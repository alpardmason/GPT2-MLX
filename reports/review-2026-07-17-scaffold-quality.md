# Review: GPT-2 MLX Learning Skeleton

**Date:** 2026-07-17  
**Branch reviewed:** `skeleton`  
**Stance:** course instructor + senior engineer  
**Scope:** scaffold quality only (no implementation of learner TODOs)

## Recommendation

**Yes — this is a strong educational scaffold.** Keep the intentional red TODO contracts, the LayerNorm scale/bias lesson, and the behavioral no-future-leakage tests. Fix the Linux CI MLX dependency before trusting GitHub Actions on `ubuntu-latest`, and loosen the causal-mask contract so additive masks (the usual GPT-2 style) are not falsely rejected.

## Summary

The project does what a learning skeleton should do: ship constructible modules, shape comments, config validation, data helpers, and contract tests, while leaving the real transformer math as large `NotImplementedError` TODOs. Pedagogy and engineering hygiene are both above average for a course repo.

Verified locally (Linux, after installing a CPU backend):

| Check | Result |
| --- | --- |
| `uv run ruff check .` | pass |
| `uv run pyright` | pass |
| `uv run pytest -m "not todo"` | 7 passed (scaffold gate) |
| `uv run pytest` | 7 passed, 12 failed with expected `NotImplementedError` |
| `python -m gpt2_mlx.train` / `generate` | hit intentional TODOs |

Without a Linux MLX backend (`mlx[cpu]` / `mlx[cuda]`), imports fail with `libmlx.so: cannot open shared object file`, which is the main infra risk for CI.

## Findings Ordered by Severity

### 1. High — Linux CI will fail to import MLX

**Where:** `pyproject.toml` depends on `mlx>=0.26.0`; `.github/workflows/ci.yml` runs on `ubuntu-latest`.

**Symptom:** On Linux, plain `mlx` does not pull a backend. Current metadata only auto-installs `mlx-metal` on Darwin; Linux needs the `cpu` or `cuda` extra (`mlx-cpu` / `mlx-cuda-*`). Collection then dies with `ImportError: libmlx.so: ...`.

**Why it matters:** The scaffold gate never actually runs on GitHub Actions Linux until this is fixed, so CI gives a false sense of coverage.

**Suggested fix:**

```toml
dependencies = [
    "mlx[cpu]>=0.26.0",
    "numpy>=2.0.0",
]
```

On Apple Silicon, `mlx` still resolves Metal; documenting `mlx[cuda]` as an optional Linux GPU path is enough for later.

**Alternatives:**

| Option | Pros | Cons |
| --- | --- | --- |
| `mlx[cpu]` (recommended) | CI-green on Ubuntu; learners can run tests on Linux too | Not the Metal path students will train on |
| macOS-only CI runner | Matches primary student hardware | Cost/availability; slower feedback for PRs |
| Skip MLX tests in CI, keep lint/pyright | Always green | Loses the LayerNorm/param-count regression signal |

**Relevant tests:** entire `pytest -m "not todo"` job.

### 2. Medium — Causal mask contract assumes a boolean allow/block mask

**Where:** `tests/test_attention_mask.py::test_causal_mask_contract`

**Issue:** The test does `bool(materialized[row, col].item())` and asserts past/self are truthy and future are falsy. That matches a boolean mask, but GPT-2-style **additive** masks use `0` for allowed and `-inf` for blocked. For additive masks:

- allowed `0.0` → `bool(0.0) is False` → false failure on the lower triangle
- blocked `-inf` → `bool(-inf) is True` → inverted semantics

The method docstring deliberately says “design and implement a causal masking strategy,” so the contract is tighter than the teaching prompt.

**Suggested fix:** Assert the *effect* of the mask on scores (e.g. after apply+softmax, upper triangle ≈ 0), or accept both conventions in the test. Prefer effect-based checks; they match `test_attention_respects_causality` better.

**Alternatives:** Keep boolean-only and state that clearly in the TODO docstring (narrower pedagogy, fewer surprises).

### 3. Medium — `dropout` is configured but never wired into modules

**Where:** `GPTConfig.dropout`; `MLP.__call__` TODO mentions “configured dropout behavior”; `CausalSelfAttention` / `MLP` `__init__` never create `nn.Dropout`.

**Issue:** Learners must invent dropout placement without a construction hook. Easy to forget attention residual dropout or to sprinkle it inconsistently.

**Suggested fix (scaffold-only):** Construct `self.dropout = nn.Dropout(config.dropout)` (and optionally attn/resid dropouts) in `__init__`, leave application inside the TODO forward methods. Or remove dropout from the pass-1 TODO text and treat it as a later extension.

### 4. Low — Split ownership of training loss

**Where:** `GPT.loss` and `train.compute_loss` are both TODOs.

**Issue:** Slight ambiguity about whether CLI loss is a thin wrapper or a second implementation. `generate_text` already delegates cleanly to `GPT.generate`.

**Suggested fix:** Make `compute_loss` docstring say “delegate to `model.loss`” (or implement that one-liner in the skeleton) so there is a single learning target.

### 5. Low — Generation test name overclaims “greedy”

**Where:** `tests/test_generation.py::test_greedy_generation_deterministic`

**Issue:** It only checks seed reproducibility at default `temperature=1.0`. That is a good contract, but “greedy” usually means argmax / `temperature → 0`.

**Suggested fix:** Rename to `test_generation_deterministic_with_seed`, or pass an explicit greedy path once you add a sampling API.

### 6. Low / note — Default config is full GPT-2 scale

**Where:** `GPTConfig` defaults (`vocab_size=50257`, `block_size=1024`, `n_layer=12`, …)

**Issue:** Fine as a realism anchor, but constructing the default model on smaller Macs is heavy for early experiments. `GPTConfig.tiny()` already protects tests.

**Suggested fix:** README note: “use `.tiny()` until forward/loss/generate work; only then try larger configs.”

## Correctness and Shape Analysis

What the scaffold gets right:

- Clear shape contracts: attention/MLP/block `[B,T,C]→[B,T,C]`, GPT `[B,T]→[B,T,V]`, loss scalar, generate `[B,T]→[B,T+N]`.
- Config validates `n_embd % n_head == 0` and dropout range.
- `make_lm_batch` encodes the classic next-token shift (`y` is `x` shifted by one) and tests it.
- LayerNorm uses `bias=config.bias` (keeps scale) with a dedicated regression test — this is exactly the pitfall called out in `AGENTS.md`.
- Behavioral causality tests (`test_no_future_leakage`, `test_attention_respects_causality`) catch leaky attention that shape tests miss.
- Generation includes a `block_size` cropping contract — good prophylaxis against the uncropped-context pitfall.
- Loss sanity band around `ln(V)` rejects “return a scalar mean” fakes.

Residual learner risks (expected until TODOs are filled):

- Boolean vs additive mask semantics (amplified by finding #2).
- Head reshape swapping `T` and `H`.
- Residual / pre-norm placement in `Block`.
- Off-by-one in loss if someone re-shifts targets that `make_lm_batch` already shifted.

## Efficiency and Resource Analysis

Appropriate for pass 1: no KV cache, no HF weight loading, no distributed training. That keeps cognitive load on architecture.

Watch-outs for learners later (not defects today):

- Materializing full `[B,H,T,T]` attention scores dominates memory at longer contexts.
- Default 1024 context × 12 layers is a steep jump from `tiny()`; stay small until the model runs.
- Regenerating a causal mask every forward is fine for learning; caching a triangular mask is a later micro-optimization.

## Tests to Add or Update

After the CI fix, consider:

1. **Mask effect test** (replace or complement boolean checks): apply mask to ones scores, softmax, assert future mass ~ 0.
2. **Dropout construction smoke test** (if you wire Dropout in `__init__`): modules exist and respect `dropout=0`.
3. **Optional:** `test_make_lm_batch` error paths (`batch_size<=0`, too-short token stream) — validation exists, coverage does not.
4. Keep TODO markers on learner contracts; do not convert them to `xfail` (matches the project TDR).

## Local-Hardware Guidance

Reference machine from project standards: Apple Silicon M3 Pro, 36 GB unified memory.

- Start with `GPTConfig.tiny()` for every new concept.
- After attention+block+GPT forward work, try a “small” config (e.g. `n_layer=4`, `n_embd=256`, `block_size=128`) before full GPT-2 defaults.
- Log approximate peak memory and tokens/sec once generate/train loops exist; that turns the toy into engineering practice.

## Production Bridge and Further Reading

Useful later, not for pass 1:

- KV cache for decode latency
- Activation checkpointing vs recompute
- Mixed precision / numerics
- Weight tying and tokenizer/checkpoint compatibility
- Eval harnesses and regression sets for generation quality

References:

- [MLX install docs](https://ml-explore.github.io/mlx/build/html/install.html) (macOS Metal vs Linux `mlx[cpu]` / `mlx[cuda]`)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- Karpathy / nanoGPT-style materials for GPT-2 block structure and LM loss alignment
- Vaswani et al., *Attention Is All You Need* (scaled dot-product attention + masking)

## Benchmarks and Evaluation Notes

Not applicable yet: forward, loss, and generate are unfinished. Once they exist, capture:

- peak memory on tiny vs small configs
- tokens/sec for generate
- max context that fits locally
- qualitative generation sanity (repetition, collapse, temperature sensitivity)

## What to Learn Next

Suggested order for filling TODOs:

1. Causal mask + `CausalSelfAttention.__call__` (shapes, then no-future-leakage)
2. MLP forward (GELU / GPT-2 activation choice)
3. `Block` pre-norm residuals
4. `GPT.__call__` embeddings → blocks → `ln_f` → `lm_head`
5. `GPT.loss` next-token cross-entropy (no extra shift if using `make_lm_batch`)
6. `GPT.generate` with context crop to `block_size`
7. Optional weight tying (`tie_weights`)

Study focus tied to this review:

- Mask conventions (boolean vs additive) and softmax numerical stability
- MLX packaging extras for cross-platform CI
- Pre-norm residual transformer block dataflow
- Why behavioral contracts beat shape-only tests for causality

---

## Rationale (why this verdict)

The skeleton preserves learner ownership of core concepts, gates CI on scaffold tests via `@pytest.mark.todo`, and already encodes hard-won pitfalls (LayerNorm scale, future leakage, context cropping). Those choices match professional teaching practice better than a completed toy model would.

## Alternatives (project direction)

| Direction | When to choose |
| --- | --- |
| Keep scaffold + fix CI/mask contract | Default: best learning ROI |
| Fill TODOs for the learner | Only if the student explicitly asks for a worked solution |
| Expand into KV cache / HF load / full trainer now | Scope creep; defer until core contracts are green |
