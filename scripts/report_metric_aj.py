"""Print one comparison line for an eval dir, against the bars this line of work must clear."""

import json
import sys
import pathlib

BARS = {"scale-refiner-alone": 0.2385, "DELTA": 0.2270, "v63": 0.2238}

for d in sys.argv[1:]:
    p = pathlib.Path(d) / "metrics.json"
    if not p.exists():
        print(f"[report] {d}: no metrics.json")
        continue
    m = json.loads(p.read_text())
    ps, mean = m["per_subset"], m["overall"]["metric_average_jaccard"]
    per = " ".join(
        f"{s}={ps[s]['metric_average_jaccard']:.4f}"
        for s in ("drivetrack", "pstudio", "adt")
        if s in ps
    )
    verdict = " ".join(
        f"{k}({v:.4f}):{'BEAT' if mean > v else 'below'}" for k, v in BARS.items()
    )
    print(f"[report] {d}  {per}  mean={mean:.4f}  |  {verdict}")
