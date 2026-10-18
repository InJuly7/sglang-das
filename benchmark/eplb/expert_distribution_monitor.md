# Decode expert distribution monitor

Independent of the EPLB recorder. Supports static DeepEP low-latency deployments
with TP=DP=EP, PP=CP=1 and one tokenizer worker. EP size is read at runtime.

Set on **all nodes** before launching the server:

```bash
export SGLANG_EXPERT_DIST_ENABLE=1
export SGLANG_EXPERT_DIST_DURATION_SEC=60
export SGLANG_EXPERT_DIST_DELAY_SEC=10
export SGLANG_EXPERT_DIST_OUTPUT_DIR=/tmp/expert-stats
```

The first generation request after readiness arms one window across ranks.
Set benchmark warmup requests to zero; keep traffic running for delay + duration.
Query `GET /expert_distribution_monitor` for per-rank state and output paths.
Each rank writes `<output_dir>/<run_id>/rank-<global_rank>.json` at expiry,
even if traffic has stopped. Collect all files in one directory, then run:

```bash
python -m sglang.srt.observability.expert_distribution_summary /tmp/expert-stats/RUN_ID
```

Outputs: logical experts, physical slots, rank load CSVs and a checked summary.
Counters represent target decode/verify token-expert assignments, including
rejected speculative candidates. Draft/shared experts and padded tokens are
excluded. DP-idle receivers still count work dispatched from active peers.
Avoid health-check generation during measurement; receiver counts describe all
work dispatched during the window. Fake-prefill results describe that synthetic
workload, not real-prefill end-to-end routing.

Graph capture includes scratch writes before the window starts. Whole forwards
are admitted by submission time. Counting stays captured after expiry; disable
via the startup flag for an uninstrumented performance comparison. No new
per-step collectives are added; EPLB start/stop/dump never resets these counters.

## DeepGEMM input dump

Also set `SGLANG_EXPERT_DIST_DUMP_GEMM=1`. The first measured forward is sampled
by default; `SGLANG_EXPERT_DIST_GEMM_MAX_STEPS=N` increases this bounded sample.
TBO/SBO must be disabled. Dumps use the final library arguments of both
`m_grouped_w4a8_gemm_nt_masked` and its HIPC variant, tagged `gate_up` / `down`.
`rank-<rank>.gemm_inputs.jsonl` records input/scales/packed weight/output/mask
shape, dtype, device, stride, contiguity, expected_m, E/M/N/K, valid/padded rows,
per-expert counts, and physical/logical IDs. The masked API has `masked_m_values`
instead of the contiguous API's `m_indices_runs` (reported as null).

Captured GPU snapshots select the variant actually replayed; no CPU reads occur
inside capture. Sampling does synchronize outside the graph and adds overhead.
Whole-window `gemm_physical_count[layer][stage][slot]` is accumulated separately
for each GEMM stage and checked against dispatch counts in
`gemm_vs_dispatch_match`. Never add gate/up and down counts together when
comparing expert assignments: both GEMMs process the same routed tokens.
