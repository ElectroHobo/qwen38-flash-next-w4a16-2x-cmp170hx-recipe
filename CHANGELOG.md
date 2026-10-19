# Changelog

## 0.1.0 — 2026-10-19

First release. Descends from Morrowmake's Qwen3.8-Flash-Next EXL3 recipe at
commit `207867e` (release 1.0.0) and runs their engine image unchanged.

- Checkpoint: `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` (AutoRound INT4
  experts, INT8 GDN projections, FP8 n-gram table, INT4-packed MTP head) at
  `cec53e9f`, instead of the EXL3 5.05 bpw build.
- Three lines in `serve.sh`: `--load-format auto`, the n-gram table offloaded
  to host RAM (`cpu_offload: true`), and `GRAPH_MODE` accepting and defaulting
  to `FULL_AND_PIECEWISE`, so the whole decode step runs as one CUDA graph.
  On a dual E5-2697A v4 (Broadwell) host this takes the step from 44 ms to
  15 ms: 206.4 / 245.6 / 186.2 tok/s for one user (structured / code / prose),
  820.7 / 826.6 / 692.4 across eight, 4,527 tok/s cold prefill, a
  1,489,261-token pool at 262,144 context.
- KV offload (`KV_OFFLOAD_GB`, `KV_OFFLOAD_DIR`), carried over from the MiMo
  recipe on the same engine: a 100k-token context evicted by other traffic
  reloads from a 64 GiB RAM tier in 0.68 s (22 s re-read without it), from an
  NVMe tier in 1.85 s; ~30 KB of KV per token.
- Quality: HumanEval 161/164, GSM8K 199/200 (first 200), thinking on, no reply
  cut off.
- Benchmarks in `bench/` (decode and prefill under the MiaAI-Lab protocol,
  engine-step probe, KV offload, KV dtype and long context, prefill fairness,
  agent probes, quality, soak) with the raw files of every run in
  `bench/results/release-0.1.0/`.
- Documented and not default: FP8 KV cache (+73 % pool, −8 to −10 % decode,
  answers shift); draft count 1 / 2 / 3; the things that did not help
  (expert sharding is impossible on this checkpoint, clock locks are
  power-bound, `NCCL_PROTO=LL` slows prefill).
- Removed Morrowmake's `tools/` (their release tooling). `docker/` is kept as
  they ship it, with the checkpoint name and pin updated in `prepare.sh` and
  `container.env`.
