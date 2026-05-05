from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.ultraleap_adapter import UltraleapTrackingAdapter


def main() -> None:
    print("Ultraleap diagnostic")
    print("Put your hand above the Ultraleap sensor for 5 seconds.")
    print()

    valid = 0
    total = 0
    with UltraleapTrackingAdapter() as tracker:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = tracker.process()
            total += 1
            if frame.is_valid:
                valid += 1
                wrist = frame.landmarks[0]
                print(
                    "HAND",
                    frame.hand_label,
                    f"points={len(frame.landmarks)}",
                    f"wrist=({wrist.x:.2f}, {wrist.y:.2f}, {wrist.z:.2f})",
                    f"fps={frame.framerate}",
                )
            time.sleep(0.05)

    print()
    print(f"valid frames: {valid}/{total}")
    if valid == 0:
        print("No hand was reported by LeapC. Check Ultraleap Control Panel visualizer.")


if __name__ == "__main__":
    main()
