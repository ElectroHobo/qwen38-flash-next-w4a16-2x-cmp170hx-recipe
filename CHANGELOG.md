# Changelog

## 0.2.0 — 2026-10-24

Many users. The repository is renamed from `qwen38-flash-next-w4a16-cmp170hx-recipe`
to `qwen38-flash-next-w4a16-2x-cmp170hx-recipe` (GitHub redirects the old name), and
has a one-card sibling, `qwen38-flash-next-3bpw-1x-cmp170hx-recipe`, on the same
engine, patches and n-gram table. Same engine image and checkpoint as 0.1.0.

- **RecoverSSM for the GDN layers** (`patches/gdn_recover/` and three edited engine
  files, on by default, `RECOVER_SSM=0` to turn off): one recurrent state per
  request instead of one per draft position, the accepted tokens replayed after
  sampling, the design Morrowmake's engine uses for GLM's KDA layers. With 48
  requests allowed, 28 fitted before; now 128 fit at 94 % of the pool. The verify
  is the engine's fused CUDA kernel recording instead of storing states, built at
  first start (~95 s) with the engine's own fast-math flags; checked bitwise
  against the original path.
- **BF16 GDN states** (`SSM_DTYPE`, default `bfloat16`; the checkpoint asks for
  float32) with 912-token blocks (`BLOCK_SIZE`; 1680 with float32): half the state
  pages, steps 1–3 % shorter at load.
- **Up to 128 requests** (`MAX_SEQS`, was 8); the DFlash 2 schedule adds 2 drafts
  from 33 requests (`[[1,2,7],[3,4,5],[5,32,3],[33,128,2]]`).
- **MTP for many users**: `SPEC_MODE=mtp` now drafts 3 / 2 / 1 by load
  (`[[1,4,3],[5,8,2],[9,128,1]]`; each MTP draft is a pass of its own) and is ahead
  of DFlash 2 from about 32 users. From a fresh clone (`bench/concurrency_sweep.py`,
  700 tokens, temperature 0): DFlash 2 216 / 772 / 1,357 / 1,937 tok/s at 1 / 8 /
  32 / 128 users, MTP 198 / 744 / 1,379 / 1,966.
- **One n-gram table for several servers** (`PLE_SHARED_DIR`): the table becomes a
  shared mapping of one file on host tmpfs, registered with CUDA, each card mapping
  its rows, so servers of this recipe and of the one-card recipe share one 48 GiB
  copy (each pins 64 GiB otherwise).
- **Copy-on-write in chunks** (`patches/v1_worker_utils.py`): a burst of partial
  prefix-cache hits gathered 4 GiB at once and stopped a one-card engine with
  `torch.OutOfMemoryError`; the copy now runs in 256 MiB chunks.
- **`BLOCK_SIZE` reaches the server** (it was read by `serve.sh` but never passed
  into the container), with the new settings.
- The patched engine files are mounted in every speculation mode now, not only
  dflash.
- `bench/concurrency_sweep.py`: aggregate decode and step counters from 1 to 128
  users.
- Numbers from a fresh clone: one user 241.2 / 349.7 / 198.4 tok/s, eight 894.5 /
  986.8 / 769.7, code editing 495, cold prefill 4,696, pool 1,475,836 tokens;
  HumanEval 162/164, GSM8K 198/200 (MTP: 162/164, 197/200). Raw files in
  `bench/results/release-0.2.0/`.

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
- DFlash 2 by default: inco.ai's `Qwen3.8-Flash-Next-DFlash2` drafter
  (Apache-2.0, `a22a1744`) downloaded and pinned beside the checkpoint, and two
  engine files in `patches/` (48 lines added, 1 changed) mounted over the image's
  so the model hands the drafter the hidden states it was trained on. 7 drafts
  for one or two requests, 5 for three or four, 3 beyond, with `--block-size
  1680` so five to seven drafts start. Against MTP: one user 239 / 353 / 204
  tok/s (+16 / +43 / +10 %), code editing 496 (+80 %), eight users 847 / 986 /
  728; HumanEval 163/164, GSM8K 197/200. `SPEC_MODE=mtp` runs the checkpoint's
  MTP head on the image as released.
- KV offload (`KV_OFFLOAD_GB`, `KV_OFFLOAD_DIR`), carried over from the MiMo
  recipe on the same engine: a 100k-token context evicted by other traffic
  reloads from a 64 GiB RAM tier in 0.68 s (22 s re-read without it), from an
  NVMe tier in 1.85 s; ~30 KB of KV per token.
- Quality with MTP: HumanEval 161/164, GSM8K 199/200 (first 200), thinking on, no reply
  cut off.
- Benchmarks in `bench/` (decode and prefill under the MiaAI-Lab protocol,
  engine-step probe, KV offload, KV dtype and long context, prefill fairness,
  agent probes, quality, soak) with the raw files of every run in
  `bench/results/release-0.1.0/`.
- Documented and not default: FP8 KV cache (+73 % pool, −8 to −10 % decode,
  answers shift); MTP and DFlash draft counts; the fair-chunk flag; the things
  that did not help
  (expert sharding is impossible on this checkpoint, clock locks are
  power-bound, `NCCL_PROTO=LL` slows prefill).
- Removed Morrowmake's `tools/` (their release tooling). `docker/` is kept as
  they ship it, with the checkpoint name and pin updated in `prepare.sh` and
  `container.env`.
