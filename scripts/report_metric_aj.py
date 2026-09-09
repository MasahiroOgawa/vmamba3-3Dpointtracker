"""Print one comparison line for an eval dir, against the bars this line of work must clear."""

import json
import sys
import pathlib

# Every bar is a measured eval dir, not an estimate. 0.2385 stood here for
# "scale-refiner-alone" and was never measured -- the figure came from the DA3-l era and the
# corrected DA3-g measurement (v89) is 0.2100. DELTA was rounded to 0.2270; it is 0.2274.
BARS = {
    "v89-refiner-alone": 0.2100,   # result/v89_scale_only/eval
    "v63": 0.2238,                 # result/v63_waft_live_2pool/eval
    "DELTA": 0.2274,               # result/20260715-1443_da3g_delta
}
# v63 run to run: 0.2238 (seed 1) vs 0.2182 (seed 2, result/v63_seed2/eval). Any gap under ~0.0056
# is inside the seed-to-seed spread of a single arm and must not be read as an effect.
#
# THIS DOES NOT APPLY TO A PAIRED COMPARISON. Two arms that share the same trained weights and
# differ only in a module -- v63 seed 1 (0.2238) against the v90 warm start (0.2191), whose v35
# tensors are bit-identical -- hold the seed fixed on both sides, and the scoring path has no
# sampling, so their difference carries no seed variance at all. Applying this bar there once
# retracted a correct measurement.
NOISE = 0.0056

if __name__ == "__main__":
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
        near = [k for k, v in BARS.items() if abs(mean - v) < NOISE]
        caveat = f"  [within seed noise {NOISE} of: {', '.join(near)}]" if near else ""
        print(f"[report] {d}  {per}  mean={mean:.4f}  |  {verdict}{caveat}")
