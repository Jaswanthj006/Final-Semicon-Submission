# Final Semicon Submission — Phase 2 & Phase 3

Repo layout:

```text
phase_2/   register.py, localize.py, train.py, model/verifier.pt, requirements.txt
phase_3/   phase3.py, cad_render.py, localize.py, train.py, requirements.txt
```

Both phases write the **same** CSV:

```text
pair_id,x,y,theta,scale,found,score
```

If `found=0`, then `x,y,theta,scale` are all `0`. One row per input pair.

---

## Install (Linux / macOS / Windows)

Use **64-bit Python 3.10–3.12** (3.11 recommended). Install **inside each folder** separately.

**Linux / macOS**

```bash
cd phase_2   # or phase_3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd phase_2   # or phase_3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Phase 3 also needs **gdstk** (GDS read); wheels exist for Windows `win_amd64`.

---

## Phase 2 — SEM reference in SEM search

**Task:** 1000×1000 reference PNG + 1000×1000 search PNG → pose + **`found`**.  
Credit uses 1 / 2 / 3 / 5 px buckets; **`found=0` on a true present zeros that row**.

**Run**

```bash
cd phase_2
python register.py --input /path/to/pairs.csv --output predictions.csv
```

**What we added vs Phase 1**

| Item | Change |
|------|--------|
| Entry | `register.py` → `predictions.csv` |
| Pose grid | scale **8–12** (Δ0.5), θ **±5°** (Δ1°) → 99 maps |
| Speed | Half-res ZNCC → top-3 poses → full-res |
| Ranking | CNN **`model/verifier.pt`** over ZNCC peaks |
| Refine | Subpixel + local θ/scale + ECC (shift cap 2.5 px) |
| `found` | Intensity + edge peaks **agree ≤3 px** and **margin ≥0.05** (not height ≥0.55) |
| Robustness | Weights path next to script; per-pair `try/except` |

**Failures → fixes (short)**

1. **Weights missing when CWD ≠ repo** → load `verifier.pt` beside `localize.py`.  
2. **One bad image killed the run** → every pair wrapped; failed row → `found=0`.  
3. **DRAM lattice + narrow pose** → widened grid + verifier.  
4. **99 maps too slow** → coarse-to-fine (~0.57 s/pair CPU on held-out test).  
5. **Height gate false reject/accept on Set B/C** → uniqueness gate above.  
6. **Set B 1–2 px x error (labels)** → generator GT uses mean row-shift over patch (eval only; matcher unchanged).

**Results — held-out `phase2_test_set` (450 pairs, present rows with `found=1`)**

| Pool | n | p@1 px | p@2 px | p@3 px | p@5 px* | Median err |
|------|--:|-------:|-------:|-------:|--------:|-----------:|
| All present (scored loc) | 348 | 0.89 | 0.99 | 1.00 | ~1.00 | 0.3 px |
| Set A | 176 | 0.97 | 1.00 | 1.00 | ~1.00 | 0.3 px |
| Set B | 172 | 0.81 | 0.98 | 0.99 | ~1.00 | 0.4 px |

\*p@5 not always printed in summary tables; with p@3 at 1.00 on All, p@5 matches for this split.

**Rejection (Set C absent):** F1 **0.965** (85.6% absent rejected; 3.3% present false-reject).

**Local CPU check (25-pair synth, organizer command):** ~**0.67 s/pair**.

---

## Phase 3 — GDS reference in SEM search

**Task:** `reference.gds` + search SEM PNG + **`search.gds`** → same CSV.  
Blind run: `reference_sem_path` / `params_json_path` **empty** (ignored).

**Run**

```bash
cd phase_3
python phase3.py --input /path/to/pairs.csv --output predictions.csv
```

**What we added**

| Item | Change |
|------|--------|
| `cad_render.py` | Yield paint GDS → grayscale template; CAD–CAD **`found`** |
| Presence | **Not** SEM softmax — ZNCC ≥0.80 + min-MAD vs runner-up gap |
| Pose | **Sobel 2θ** ZNCC (edges), scale **10×**, θ **±10°**, no `verifier.pt` |
| Prior | CAD–CAD site → `propose_near_prior` (no wrong DRAM cell) |
| FOV | Partial template if ≥**40%** overlap; refine cap **5 px** |
| Missing `search.gds` | `found=0` (no SEM fallback) |

**Failures → fixes (short)**

1. **GDS has no brightness** → yield raster + beam blur in `cad_render.py`.  
2. **Same-pitch decoys ~0.93 ZNCC** → MAD + second-peak gap (not MAD≤5 alone).  
3. **Eval looked like matcher x-bug** → generator GT fixed with **`drifted_xy`** (not in `phase3.py`).  
4. **FOV corners jumped lattice** → partial overlap matching.  
5. **Grey ZNCC on noisy SEM** → 2θ orientation maps.  
6. **One bad pair / Windows paths** → same pattern as Phase 2.

**Results — local synth (not fitted; new seeds)**

| Split | Present | p@1 | p@2 | p@3 | p@5 | Notes |
|-------|--------:|----:|----:|----:|----:|-------|
| 100 DRAM (seed 20260918199) | 76 | 94.7% | ~100% | ~100% | **100%** | CAD F1 **1.0**, med **0.37 px** |
| 150 present | 123 | 95.3% | ~100% | ~100% | 98.7% | med **0.42 px** |
| 150 + absent (~20%) | — | — | — | — | p@2 **100%** on present | absent rej **96.3%**, FP **0%** |

**Local CPU check (25-pair synth):** ~**0.28 s/pair**.

---

## Contact sheets (visual QA)

- **phase_2:** `dram25_contact_*.png` — GT overlays on SEM pairs.  
- **phase_3:** `phase3_contact_3x3.png` — 5 present / 4 absent (yellow = GT, orange = absent).

---

## Clone

```bash
git clone https://github.com/Jaswanthj006/Final-Semicon-Submission.git
cd Final-Semicon-Submission
```

Run commands from **`phase_2`** or **`phase_3`**, not the repo root.
