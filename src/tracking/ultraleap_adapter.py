from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Any

from src.models import Point2D, TrackingFrame


ULTRALEAP_INSTALL_HINT = (
    "Ultraleap SDK не найден.\n"
    "Установите Ultraleap Hand Tracking Software.\n"
    "Проверьте наличие LeapC.dll в C:\\Program Files\\Ultraleap\\LeapSDK\\lib\\x64."
)

LEAP_EVENT_TRACKING = 0x100
LEAP_TRACKING_MODE_DESKTOP = 0


@dataclass(frozen=True)
class UltraleapWorkspace:
    width_mm: float = 300.0
    height_mm: float = 300.0
    center_y_mm: float = 200.0

    @classmethod
    def from_env(cls) -> "UltraleapWorkspace":
        return cls(
            width_mm=float(os.getenv("ULTRALEAP_WORKSPACE_WIDTH_MM", "300")),
            height_mm=float(os.getenv("ULTRALEAP_WORKSPACE_HEIGHT_MM", "300")),
            center_y_mm=float(os.getenv("ULTRALEAP_WORKSPACE_CENTER_Y_MM", "200")),
        )


def normalize_ultraleap_vector(vector: Any, workspace: UltraleapWorkspace) -> Point2D:
    """Convert Ultraleap millimetres to normalized UI/scoring coordinates."""
    return Point2D(
        x=0.5 + float(vector.x) / workspace.width_mm,
        y=0.5 - (float(vector.y) - workspace.center_y_mm) / workspace.height_mm,
        z=float(vector.z) / workspace.width_mm,
    )


def _hand_label(raw_type: Any) -> str:
    if raw_type == 0 or "left" in str(raw_type).lower():
        return "Left"
    if raw_type == 1 or "right" in str(raw_type).lower():
        return "Right"
    return "Unknown"


def _default_leapc_paths() -> list[str]:
    paths = []
    env_path = os.getenv("ULTRALEAP_LEAPC_PATH")
    if env_path:
        paths.append(env_path)
    paths.extend([
        r"C:\Program Files\Ultraleap\LeapSDK\lib\x64\LeapC.dll",
        r"C:\Program Files\Ultraleap\LeapSDK\leapc_cffi\LeapC.dll",
        r"C:\Program Files\Ultraleap\OpenXR\LeapC.dll",
    ])
    return paths


class LEAP_VECTOR(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class LEAP_QUATERNION(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


class LEAP_BONE(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("prev_joint", LEAP_VECTOR),
        ("next_joint", LEAP_VECTOR),
        ("width", ctypes.c_float),
        ("rotation", LEAP_QUATERNION),
    ]


class LEAP_DIGIT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("finger_id", ctypes.c_int32),
        ("bones", LEAP_BONE * 4),
        ("is_extended", ctypes.c_uint32),
    ]


class LEAP_PALM(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("position", LEAP_VECTOR),
        ("stabilized_position", LEAP_VECTOR),
        ("velocity", LEAP_VECTOR),
        ("normal", LEAP_VECTOR),
        ("width", ctypes.c_float),
        ("direction", LEAP_VECTOR),
        ("orientation", LEAP_QUATERNION),
    ]


class LEAP_HAND(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("type", ctypes.c_int32),
        ("confidence", ctypes.c_float),
        ("visible_time", ctypes.c_uint64),
        ("pinch_distance", ctypes.c_float),
        ("grab_angle", ctypes.c_float),
        ("pinch_strength", ctypes.c_float),
        ("grab_strength", ctypes.c_float),
        ("palm", LEAP_PALM),
        ("digits", LEAP_DIGIT * 5),
        ("arm", LEAP_BONE),
    ]


class LEAP_FRAME_HEADER(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("reserved", ctypes.c_void_p),
        ("frame_id", ctypes.c_int64),
        ("timestamp", ctypes.c_int64),
    ]


class LEAP_TRACKING_EVENT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("info", LEAP_FRAME_HEADER),
        ("tracking_frame_id", ctypes.c_int64),
        ("nHands", ctypes.c_uint32),
        ("pHands", ctypes.POINTER(LEAP_HAND)),
        ("framerate", ctypes.c_float),
    ]


class LEAP_CONNECTION_MESSAGE(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("type", ctypes.c_int32),
        ("pointer", ctypes.c_void_p),
        ("device_id", ctypes.c_uint32),
    ]


class _LeapDeviceRef(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("handle", ctypes.c_void_p), ("id", ctypes.c_uint32)]


def _load_leapc() -> ctypes.CDLL:
    for path in _default_leapc_paths():
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError(ULTRALEAP_INSTALL_HINT)


class _LeapCBackend:
    _DEVICE_EVENT = 3

    def __init__(self, workspace: UltraleapWorkspace):
        self._workspace = workspace
        self._dll = _load_leapc()
        self._connection = ctypes.c_void_p()
        self._device = ctypes.c_void_p()
        self._opened = False
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._dll.LeapCreateConnection.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.LeapCreateConnection.restype = ctypes.c_int32
        self._dll.LeapOpenConnection.argtypes = [ctypes.c_void_p]
        self._dll.LeapOpenConnection.restype = ctypes.c_int32
        self._dll.LeapSetTrackingMode.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self._dll.LeapSetTrackingMode.restype = ctypes.c_int32
        self._dll.LeapPollConnection.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(LEAP_CONNECTION_MESSAGE),
        ]
        self._dll.LeapPollConnection.restype = ctypes.c_int32
        self._dll.LeapGetDeviceList.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
        ]
        self._dll.LeapGetDeviceList.restype = ctypes.c_int32
        self._dll.LeapOpenDevice.argtypes = [_LeapDeviceRef, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.LeapOpenDevice.restype = ctypes.c_int32
        self._dll.LeapSubscribeEvents.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._dll.LeapSubscribeEvents.restype = ctypes.c_int32
        self._dll.LeapCloseDevice.argtypes = [ctypes.c_void_p]
        self._dll.LeapCloseDevice.restype = None
        self._dll.LeapCloseConnection.argtypes = [ctypes.c_void_p]
        self._dll.LeapCloseConnection.restype = None
        self._dll.LeapDestroyConnection.argtypes = [ctypes.c_void_p]
        self._dll.LeapDestroyConnection.restype = None

    def open(self) -> None:
        result = self._dll.LeapCreateConnection(None, ctypes.byref(self._connection))
        if result != 0:
            raise RuntimeError(f"LeapCreateConnection failed: {result}")
        result = self._dll.LeapOpenConnection(self._connection)
        if result != 0:
            raise RuntimeError(f"LeapOpenConnection failed: {result}")
        self._dll.LeapSetTrackingMode(self._connection, LEAP_TRACKING_MODE_DESKTOP)
        self._opened = True
        self._wait_for_device_and_subscribe(timeout_sec=3.0)

    def _wait_for_device_and_subscribe(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            msg = LEAP_CONNECTION_MESSAGE()
            msg.size = ctypes.sizeof(LEAP_CONNECTION_MESSAGE)
            if self._dll.LeapPollConnection(self._connection, 100, ctypes.byref(msg)) != 0:
                continue
            if msg.type != self._DEVICE_EVENT:
                continue
            count = ctypes.c_uint32(0)
            self._dll.LeapGetDeviceList(self._connection, None, ctypes.byref(count))
            if count.value == 0:
                continue
            DeviceRefArray = _LeapDeviceRef * count.value
            dev_array = DeviceRefArray()
            self._dll.LeapGetDeviceList(self._connection, dev_array, ctypes.byref(count))
            if self._dll.LeapOpenDevice(dev_array[0], ctypes.byref(self._device)) == 0:
                self._dll.LeapSubscribeEvents(self._connection, self._device)
            return

    def poll(self, timeout_ms: int) -> TrackingFrame:
        message = LEAP_CONNECTION_MESSAGE()
        message.size = ctypes.sizeof(LEAP_CONNECTION_MESSAGE)
        result = self._dll.LeapPollConnection(self._connection, timeout_ms, ctypes.byref(message))
        now = time.monotonic()

        if result != 0 or message.type != LEAP_EVENT_TRACKING or not message.pointer:
            return TrackingFrame(
                timestamp=now,
                landmarks=[],
                is_valid=False,
                source="ultraleap",
                coordinate_system="image_normalized",
            )

        event = ctypes.cast(message.pointer, ctypes.POINTER(LEAP_TRACKING_EVENT)).contents
        if event.nHands < 1 or not event.pHands:
            return TrackingFrame(
                timestamp=now,
                landmarks=[],
                is_valid=False,
                source="ultraleap",
                coordinate_system="image_normalized",
                framerate=float(event.framerate),
            )

        return self._hand_to_frame(event.pHands[0], event, now)

    def _hand_to_frame(
        self,
        hand: LEAP_HAND,
        event: LEAP_TRACKING_EVENT,
        timestamp: float,
    ) -> TrackingFrame:
        landmarks = [normalize_ultraleap_vector(hand.arm.next_joint, self._workspace)]

        for digit in hand.digits:
            proximal = digit.bones[1]
            intermediate = digit.bones[2]
            distal = digit.bones[3]
            for vector in (
                proximal.prev_joint,
                proximal.next_joint,
                intermediate.next_joint,
                distal.next_joint,
            ):
                landmarks.append(normalize_ultraleap_vector(vector, self._workspace))

        return TrackingFrame(
            timestamp=timestamp,
            landmarks=landmarks,
            is_valid=len(landmarks) == 21,
            hand_label=_hand_label(hand.type),
            source="ultraleap",
            coordinate_system="image_normalized",
            confidence=float(hand.confidence),
            framerate=float(event.framerate),
        )

    def close(self) -> None:
        if self._device:
            self._dll.LeapCloseDevice(self._device)
            self._device = ctypes.c_void_p()
        if self._connection:
            if self._opened:
                self._dll.LeapCloseConnection(self._connection)
                self._opened = False
            self._dll.LeapDestroyConnection(self._connection)
            self._connection = ctypes.c_void_p()


class UltraleapTrackingAdapter:
    source_name = "ultraleap"
    model_name = "Ultraleap LeapC"
    model_sha256 = None
    requires_video = False

    def __init__(
        self,
        workspace: UltraleapWorkspace | None = None,
        poll_timeout_ms: int = 35,
    ):
        self._workspace = workspace or UltraleapWorkspace.from_env()
        self._poll_timeout_ms = poll_timeout_ms
        self._backend = _LeapCBackend(self._workspace)

    def process(self, bgr_frame: Any | None = None) -> TrackingFrame:
        return self._backend.poll(self._poll_timeout_ms)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "UltraleapTrackingAdapter":
        self._backend.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
