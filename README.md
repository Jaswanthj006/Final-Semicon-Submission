# Final Semicon Submission — Phase 2 & Phase 3

Pattern registration and localization for SEM/GDS pairs, split into two phases:

- **Phase 2** — SEM reference in SEM search
- **Phase 3** — GDS reference in SEM search

Both phases write the **same** output CSV schema:

```text
pair_id,x,y,theta,scale,found,score
```

If `found=0`, then `x`, `y`, `theta`, `scale` are all `0`. One row is written per input pair.

## Repository layout

```text
phase_2/
├── register.py        # entry point
├── localize.py
├── train.py
├── model/verifier.pt
└── requirements.txt

phase_3/
├── phase3.py           # entry point
├── cad_render.py
├── localize.py
├── train.py
└── requirements.txt
```

## Clone

```bash
git clone https://github.com/Jaswanthj006/Final-Semicon-Submission.git
cd Final-Semicon-Submission
```

Run all commands from **`phase_2/`** or **`phase_3/`** — not the repo root.

## Installation

Requires **64-bit Python 3.10–3.12** (3.11 recommended). Install dependencies **separately inside each phase folder**.

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
cd phase_2   # or phase_3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
cd phase_2   # or phase_3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

</details>

> Phase 3 additionally requires **gdstk** (GDS reading). Prebuilt wheels are available for `win_amd64`.

---

## Phase 2 — SEM reference in SEM search

**Task:** given a 1000×1000 reference PNG and a 1000×1000 search PNG, predict pose and presence (`found`).

Credit is scored at 1 / 2 / 3 / 5 px buckets. A true-present pair scored with `found=0` zeros that row.

### Run

```bash
cd phase_2
python register.py --input /path/to/pairs.csv --output predictions.csv
```

### Changes vs. Phase 1

| Item | Change |
|---|---|
| Entry point | `register.py` → `predictions.csv` |
| Pose grid | scale **8–12** (Δ0.5), θ **±5°** (Δ1°) → 99 maps |
| Speed | Half-res ZNCC → top-3 poses → full-res refinement |
| Ranking | CNN **`model/verifier.pt`** scores ZNCC peaks |
| Refinement | Subpixel + local θ/scale search + ECC (shift capped at 2.5 px) |
| `found` gate | Intensity and edge peaks agree within **3 px** and margin **≥ 0.05** (replaces the old height ≥ 0.55 gate) |
| Robustness | Weights loaded relative to script path; per-pair `try/except` |

### Issues found and fixed

1. **Weights not found when CWD ≠ repo root** — now loads `verifier.pt` relative to `localize.py`.
2. **One bad image failed the whole run** — every pair is now wrapped in `try/except`; a failed pair returns `found=0` instead of crashing.
3. **DRAM lattice caused pose ambiguity** — widened the pose search grid and added the verifier network.
4. **99 candidate maps were too slow** — switched to coarse-to-fine search (~0.57 s/pair on CPU, held-out test).
5. **Height-based gate false-rejected/accepted on Sets B/C** — replaced with the intensity/edge agreement + margin gate above.
6. **Set B showed a 1–2 px x-error** — traced to label generation, not the matcher; ground truth now uses mean row-shift over the patch (eval-only fix, matcher unchanged).

### Results — held-out `phase2_test_set` (450 pairs)

Localization precision on present rows with `found=1`:

| Pool | n | p@1px | p@2px | p@3px | p@5px* | Median error |
|---|---:|---:|---:|---:|---:|---:|
| All present | 348 | 0.89 | 0.99 | 1.00 | ~1.00 | 0.3 px |
| Set A | 176 | 0.97 | 1.00 | 1.00 | ~1.00 | 0.3 px |
| Set B | 172 | 0.81 | 0.98 | 0.99 | ~1.00 | 0.4 px |

\* p@5 isn't always reported directly; since p@3 = 1.00 on the "All" pool, p@5 matches for this split.

**Presence rejection (Set C, absent pairs):** F1 = **0.965** — 85.6% of absent pairs correctly rejected, 3.3% false-reject rate on present pairs.

**Local CPU benchmark** (25-pair synthetic set, organizer command): ~**0.67 s/pair**.

---

## Phase 3 — GDS reference in SEM search

**Task:** given `reference.gds`, a search SEM PNG, and `search.gds`, predict the same CSV schema as Phase 2.

In a blind run, `reference_sem_path` and `params_json_path` are empty and ignored.

### Run

```bash
cd phase_3
python phase3.py --input /path/to/pairs.csv --output predictions.csv
```

### Changes

| Item | Change |
|---|---|
| `cad_render.py` | Renders yield-paint GDS into a grayscale template; enables CAD–CAD `found` scoring |
| Presence | Not SEM softmax — ZNCC ≥ 0.80 plus a min-MAD gap vs. the runner-up match |
| Pose search | Sobel 2θ ZNCC over edges, scale search 10×, θ **±10°**, no `verifier.pt` |
| Prior | CAD–CAD site match feeds `propose_near_prior` to avoid locking onto the wrong DRAM cell |
| Field of view | Matches partial templates when overlap is ≥ **40%**; refinement capped at 5 px |
| Missing `search.gds` | Row is scored `found=0` (no SEM-only fallback) |

### Issues found and fixed

1. **GDS has no brightness information** — `cad_render.py` adds yield-paint rasterization plus beam blur to simulate SEM contrast.
2. **Same-pitch decoys scored ~0.93 ZNCC** — added a MAD threshold combined with a second-peak gap, rather than relying on MAD ≤ 5 alone.
3. **Apparent matcher x-axis bug** — was actually a ground-truth generation issue; fixed via `drifted_xy` in the eval generator (not in `phase3.py` itself).
4. **Pose jumped to the wrong lattice cell near FOV corners** — added partial-overlap template matching.
5. **Grayscale ZNCC was unreliable on noisy SEM** — switched to 2θ orientation (Sobel edge) maps.
6. **One bad pair / Windows path handling** — same per-pair `try/except` and path-relative loading pattern as Phase 2.

### Results — local synthetic sets (unseen seeds, not fitted)

| Split | Present pairs | p@1 | p@2 | p@3 | p@5 | Notes |
|---|---:|---:|---:|---:|---:|---|
| 100 DRAM (seed 20260918199) | 76 | 94.7% | ~100% | ~100% | **100%** | CAD F1 = **1.0**, median 0.37 px |
| 150 present | 123 | 95.3% | ~100% | ~100% | 98.7% | median 0.42 px |
| 150 + ~20% absent | — | — | — | — | p@2 = **100%** on present | 96.3% absent rejection, 0% false positives |

**Local CPU benchmark** (25-pair synthetic set): ~**0.28 s/pair**.

---

## Contact sheets (visual QA)

| File | Contents |
|---|---|
| `phase_2/dram25_contact_*.png` | Ground-truth overlays on SEM pairs |
| `phase_3/phase3_contact_3x3.png` | 5 present + 4 absent pairs (yellow = ground truth, orange = absent) |
