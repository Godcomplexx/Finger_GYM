# ICF / MKF scoring

The application stores an ICF-compatible profile in `summary.icf_codes` and in
the exported JSON field `icfCodes`.

ICF codes are complete only when a qualifier is present after the dot:

| Qualifier | Meaning | Problem percent |
| --- | --- | --- |
| `0` | no problem | 0-4% |
| `1` | mild problem | 5-24% |
| `2` | moderate problem | 25-49% |
| `3` | severe problem | 50-95% |
| `4` | complete problem | 96-100% |
| `8` | not specified | insufficient information |
| `9` | not applicable | not applicable |

The internal Finger GYM score grows when the patient performs better. The ICF
problem qualifier grows when the problem is more severe. For that reason the
conversion is inverted:

```text
problem_percent = round((1 - score / max_score) * 100)
```

Then `problem_percent` is converted to the ICF qualifier by the table above.

Current exported codes:

| Code | Domain | Source |
| --- | --- | --- |
| `s110` | body structures | `.8`, not measured by webcam hand test |
| `b7302` | body functions | functional proxy from hand movement tasks |
| `d520` | activities and participation | functional proxy from total hand test score |
| `e310` | environmental factors | `.8`, not measured by webcam hand test |

Values marked as `functional_proxy` are comparable with the ICF 0-4 severity
scale, but they are not a standalone clinical diagnosis and require specialist
confirmation.
