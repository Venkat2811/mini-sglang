# H100 Closure Run Runbook

Date: 2026-05-14
Status: Draft (ready for rental, no spend yet)

This runbook brings `mini-sglang` up on a rented single-node H100 80GB and captures the closure A/B numbers per `RFC 0001`.

## Pre-flight (Local)

These steps cost nothing and can be done before any rental.

```bash
# Verify branch and clean working tree (ignore .log artifacts in baselines)
cd mini-sglang
git status --short --branch
git log --oneline -3

# Verify the closure scripts exist
ls scripts/closure/

# Verify Python toolchain matches the runbook
python3.12 --version || echo "need Python 3.12.x"
which uv || echo "need uv"
```

## Provider Bring-up (Generic)

This runbook is provider-agnostic. Concrete examples assume a Lambda, RunPod, or Vast.ai single-node H100 80GB with Ubuntu 22.04+ and CUDA 12.4+ driver pre-installed. Adapt as needed.

```bash
# 1. SSH in (provider-specific command; redact host details from any committed log)
ssh <ephemeral-host>

# 2. Clone mini-sglang on the host (do not commit any path containing the user's home)
cd /workspace
git clone --branch rust-engine-cpu-attempt-1-closure \
  https://github.com/Venkat2811/mini-sglang.git mini-sglang
cd mini-sglang

# 3. Install dependencies via uv per the existing RUNBOOK section 1
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -e .

# 4. Build the PyO3 extension for the Rust hotpath backend
uvx maturin develop --release --manifest-path rust/minisgl-cpu-py/Cargo.toml

# 5. Pre-pull the model weights to avoid clock starting on download
huggingface-cli download Qwen/Qwen3-4B --quiet || \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B')"

# 6. Sanity-check imports
.venv/bin/python -c "import torch, transformers, flashinfer, sgl_kernel, tvm_ffi, minisgl, minisgl_cpu; \
  print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); \
  print('device', torch.cuda.get_device_name(0))"
```

## ShareGPT Subset Prep

ShareGPT subset is prepared once on the host and reused across all six runs. The script ships with the repo.

```bash
.venv/bin/python scripts/closure/sharegpt_prep.py \
  --out 0_venkat-worklog/baselines/closure/sharegpt_subset.jsonl \
  --count 200 \
  --seed 42

# Confirm the subset
wc -l 0_venkat-worklog/baselines/closure/sharegpt_subset.jsonl
head -1 0_venkat-worklog/baselines/closure/sharegpt_subset.jsonl
```

If the corpus download is awkward on the rental host, the script also accepts a
locally pre-prepared `--input` JSONL. The corpus source is `anon8231489123/ShareGPT_Vicuna_unfiltered` or the cleaner `philschmid/sharegpt-raw` derivative used by similar Myelon benchmarks.

## Six Sequential Runs

Each backend runs three times. Alternate order to reduce thermal drift bias. Total runs: six. Per-run wall-clock budget: 4-6 minutes for 200 prompts at Qwen3-4B on H100.

```bash
PROMPTS=0_venkat-worklog/baselines/closure/sharegpt_subset.jsonl
OUTDIR=0_venkat-worklog/baselines/closure
MODEL=Qwen/Qwen3-4B

# Run 1: Python control
MASTER_PORT=29501 .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend python --model-path "$MODEL" --prompts "$PROMPTS" --run-index 1 \
  --out "$OUTDIR/run-python-1.json"

# Run 2: Rust hotpath
MASTER_PORT=29502 .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend rust_hotpath --model-path "$MODEL" --prompts "$PROMPTS" --run-index 1 \
  --out "$OUTDIR/run-rust-1.json"

# Run 3: Rust hotpath (re-pre-warmed)
MASTER_PORT=29503 .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend rust_hotpath --model-path "$MODEL" --prompts "$PROMPTS" --run-index 2 \
  --out "$OUTDIR/run-rust-2.json"

# Run 4: Python (alternating)
MASTER_PORT=29504 .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend python --model-path "$MODEL" --prompts "$PROMPTS" --run-index 2 \
  --out "$OUTDIR/run-python-2.json"

# Run 5: Python (third)
MASTER_PORT=29505 .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend python --model-path "$MODEL" --prompts "$PROMPTS" --run-index 3 \
  --out "$OUTDIR/run-python-3.json"

# Run 6: Rust hotpath (third)
MASTER_PORT=29506 .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend rust_hotpath --model-path "$MODEL" --prompts "$PROMPTS" --run-index 3 \
  --out "$OUTDIR/run-rust-3.json"
```

## Parity Spot-Check (One Shadow Run)

Per RFC, one shadow run validates that `rust_hotpath` is parity-correct under this workload. Not used as a benchmark.

```bash
MASTER_PORT=29510 \
MINISGL_CPU_BACKEND_SHADOW=1 \
MINISGL_CPU_BACKEND_SHADOW_REPORT="$OUTDIR/shadow-divergence.jsonl" \
MINISGL_CPU_BACKEND_SHADOW_MAX_DIFFS=256 \
  .venv/bin/python scripts/closure/run_closure_benchmark.py \
  --backend rust_hotpath --model-path "$MODEL" --prompts "$PROMPTS" --run-index shadow \
  --out "$OUTDIR/run-shadow-parity.json"

.venv/bin/python -m minisgl.benchmark.shadow_report \
  --input "$OUTDIR/shadow-divergence.jsonl" --allow-missing --top 10
```

## Side-by-Side Comparison

After all runs land:

```bash
.venv/bin/python scripts/closure/compare_runs.py \
  --python "$OUTDIR/run-python-1.json" "$OUTDIR/run-python-2.json" "$OUTDIR/run-python-3.json" \
  --rust "$OUTDIR/run-rust-1.json" "$OUTDIR/run-rust-2.json" "$OUTDIR/run-rust-3.json" \
  --out "$OUTDIR/SIDE_BY_SIDE.md"
```

## Tear Down

Provider-specific. On Lambda or RunPod use the dashboard. On Vast.ai destroy the instance via CLI or web UI. Hard-stop within the spend cap.

## Closure Commit Sequence (Back On Laptop)

```bash
# Pull retained artifacts off the rental host (rsync or scp; do not commit absolute paths)
# Then commit on closure branch
cd mini-sglang
git checkout rust-engine-cpu-attempt-1-closure
git add 0_venkat-worklog/baselines/closure/
git commit -m "closure: H100 weekend retained artifacts"
git push origin rust-engine-cpu-attempt-1-closure
```

## Stop Conditions (Re-stated From RFC)

- one weekend wall-clock cap
- three runs per backend cap
- USD `100` total compute spend cap
- one-hour single-blocker debug ceiling before falling back to retro-only closure
