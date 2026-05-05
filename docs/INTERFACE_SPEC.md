# Спецификация обмена данными

## Назначение

Документ описывает JSON-протокол, который модуль сохраняет локально и может передавать в основную систему АПК.

## Формат

```json
{
  "module": {
    "name": "Finger GYM",
    "version": "0.2.0",
    "algorithmVersion": "ruleset-2026-05-05",
    "modelName": "MediaPipe Hand Landmarker",
    "modelSha256": "...",
    "trackingSource": "mediapipe"
  },
  "sessionId": "test-save-001",
  "patientId": "patient-test",
  "hand": "right",
  "startedAt": "2026-05-05T08:00:00+00:00",
  "calibration": {
    "palmWidth": 0.15,
    "palmCenter": {"x": 0.5, "y": 0.5}
  },
  "validTrackingRatio": 0.85,
  "exercises": [],
  "blockScores": {},
  "totalScore": 75,
  "qualityCategory": "good",
  "requiresSpecialistConfirmation": true,
  "technicalValidity": {
    "isInterpretable": true,
    "reason": null
  },
  "recommendation": {
    "mode": "standard",
    "label": "Стандартный VR-сценарий",
    "notes": []
  },
  "events": []
}
```

## Обязательные поля

- `module.version`, `module.algorithmVersion`, `module.modelSha256`, `module.trackingSource`;
- `sessionId`;
- `patientId` или обезличенный код;
- `hand`;
- `startedAt`;
- `validTrackingRatio`;
- `totalScore`;
- `qualityCategory`;
- `technicalValidity`;
- `recommendation`;
- `events`.

## Категории качества

- `good` - качество выполнения хорошее;
- `medium` - среднее качество;
- `poor` - низкое качество;
- `unreliable` - технически непригодный результат.

## Правило интерпретации

Основная система АПК не должна использовать рекомендацию без подтверждения специалистом. Если `technicalValidity.isInterpretable = false`, результат должен отображаться как технически непригодный.
