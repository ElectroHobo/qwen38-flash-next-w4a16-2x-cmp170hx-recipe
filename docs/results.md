# Results

Every number here was measured on the same box: 2× CMP 170HX (74 SMs, 64 GiB,
180 W, PCIe Gen2 x16, P2P on) in a dual Xeon E5-2697A v4 (Broadwell) host with
204 GiB of RAM, driver 610.57.04, Morrowmake's engine at
`18305948acba3f4c4d912cdd56f2202d2ebd94e8`. Raw files for each run are in
[`bench/results/release-0.1.0/`](../bench/results/release-0.1.0/); the harnesses
are in [`bench/`](../bench/). Dates are 2026-10-18 and 2026-10-19.

## Release 0.1.0

### The serving configuration

Recipe defaults: `halt95/Qwen3.8-Flash-Next-W4A16-Merlin`, TP2 with whole
experts per card, the n-gram table in host RAM, `FULL_AND_PIECEWISE` CUDA
graphs, 3 MTP drafts, bf16 KV, 262,144 context, `MAX_BATCHED` 3460.
`cmp_recipe_bench.py` (MiaAI-Lab's protocol: 400 tokens, temperature 0, thinking
off; one user the median of five streaming runs, more users the median aggregate
of three runs, prefill the median of three real-text prompts repeated three
times). Raw: `cmp-recipe-bench-merlin-morrowmake-fullgraph-1019.*`.

| | tok/s |
|---|---:|
| 1 user, structured / code / prose | 206.4 / 245.6 / 186.2 |
| 2 users, total | 327.3 / 361.9 / 277.0 |
| 4 users, total | 506.9 / 541.1 / 452.1 |
| 8 users, total | 820.7 / 826.6 / 692.4 |
| Cold prefill (21,868 / 31,224 / 34,905-token prompts) | 4,527 |
| KV pool | 1,489,261 tokens |

The engine step (`vllm_conc_steps.py`, 400 tokens pinned, from the server's own
`iteration_tokens_total_count`; raw `conc-steps-merlin-morrowmake-fullgraph-1019.txt`):
**15.2 ms at one user, 19.6 at two, 23 to 25 at four.** MTP accepts 0.84 /
0.56 / 0.45 of its three drafts by position on structured prompts and 0.92 /
0.84 / 0.79 on code, for 2.9 to 3.5 tokens per request-step.

### Why the step is 15 ms and not 44: the CPU, the table and the graph

The same checkpoint on the same engine in `PIECEWISE` graph mode (Morrowmake's
recipe default, forced by their EXL3 table reader) measured **44 ms per step**
on this host, for 71 / 83 / 63 tok/s at one user and 502 / 558 / 447 at eight
(raw `cmp-recipe-bench-merlin-morrowmake-1018.*`,
`conc-steps-merlin-morrowmake-1018.txt`). Their EXL3 recipe on this host: the
same 43 ms, 55 to 70 tok/s. The weights are not the difference; the graph is.
When the n-gram lookup is a host-side read mid-step, the step is split into
pieces and the Python between them costs ~44 ms on a 3.5 GHz Broadwell (a
current desktop CPU does the same loop in well under half the time, which is
how the EXL3 recipe reaches its published 153 tok/s on its author's box).

With the FP8 table in host RAM (`cpu_offload: true`), the engine's non-EXL3
table layer captures straight through in `FULL` mode, so `GRAPH_MODE=FULL_AND_PIECEWISE`
captures the whole decode step. That is the three-line change and the whole
reason for this recipe.

| Same checkpoint, same engine | 1 user | 8 users | prefill | step |
|---|---:|---:|---:|---:|
| `PIECEWISE` (EXL3 recipe's mode) | 71.1 / 82.6 / 63.0 | 501.9 / 558.1 / 446.9 | 4,563 | 44 ms |
| **`FULL_AND_PIECEWISE` (this recipe)** | **206.4 / 245.6 / 186.2** | **820.7 / 826.6 / 692.4** | 4,527 | **15.2 ms** |

### Against Morrowmake's EXL3 recipe

Their published figures are from their box with a current CPU; ours from the
Broadwell host. Same protocol, same card count, same context.

| | Morrowmake's EXL3 recipe (published) | This recipe |
|---|---:|---:|
| 1 user, structured / code / prose | 152.8 / 152.6 / 118.3 | 206.4 / 245.6 / 186.2 |
| 2 users, total | 236.5 / 265.0 / 204.6 | 327.3 / 361.9 / 277.0 |
| 4 users, total | 357.1 / 458.6 / 327.2 | 506.9 / 541.1 / 452.1 |
| 8 users, total | 588.2 / 674.7 / 489.9 | 820.7 / 826.6 / 692.4 |
| Cold prefill | 2,651 | 4,527 |
| KV pool at 262k | 1,044,325 | 1,489,261 |

What the comparison does not say: the EXL3 weights are 5.05 bits per weight, a
finer quantisation than INT4 experts, and no quality score has been run between
the two ([known-limits.md](known-limits.md)). The table in RAM costs 51 GB of
host memory that the EXL3 recipe does not need.

### KV offload: a RAM tier, and a disk tier behind it

`bench/kv_offload_bench.py`: a 100k-token prefix with three hidden codes is
read cold, read again warm, then 18 unique ~100k-token prompts (1.8 M tokens,
more than the pool) push it off the cards, then it is read again. Thinking off,
temperature 0. Raw `kv-offload-qwen-{off,ram64,disk}-1019.txt`.

| row (100k prefix) | no tier | `KV_OFFLOAD_GB=64` | `KV_OFFLOAD_GB=16` + NVMe `KV_OFFLOAD_DIR` |
|---|---:|---:|---:|
| cold | 21.99 s | 22.06 s | 22.00 s |
| warm (still on the cards) | 0.35 s | 0.42 s | 0.44 s |
| **evicted** by the flood | **21.98 s** (re-read) | **0.68 s** (from RAM) | **1.85 s** (from disk) |
| evicted-again (GPU hit) | 0.34 s | 0.38 s | 0.39 s |
| hidden codes retrieved | — | 3 / 3 | 3 / 3 |

The RAM run loaded 3.03 GB for the 100,224-token prefix, so this build's KV is
**about 30 KB per token**: a 64 GiB tier holds roughly 2.2 million tokens. The
disk run read 2.99 GB back from the fs tier and wrote 61 GB to it over the run.
**The disk tier is write-through and is never pruned**: one run wrote 58 GB and
took a 66 GB-free NVMe to 99 %. Put it on a volume of its own with a hard size.

### FP8 KV cache (not the default)

`QWEN_DEFAULT_KV_CACHE_DTYPE=fp8` in `serve.sh`, everything else the serving
configuration. `kv_dtype_check.py` on both servers and `cmp_recipe_bench.py`
(raw `kv-dtype-{bf16-kv-v2,fp8-kv}.json`,
`cmp-recipe-bench-merlin-morrowmake-fullgraph-fp8kv-1019.*`).

| | BF16 KV (default) | FP8 KV |
|---|---:|---:|
| KV pool at 262k | 1,489,261 | 2,578,229 (+73 %; the linear-attention state and indexer caches stay bf16) |
| 1 user, structured / code / prose | 206.4 / 245.6 / 186.2 | 186.4 / 230.2 / 170.2 |
| 8 users, total | 820.7 / 826.6 / 692.4 | 755.2 / 785.4 / 648.1 |
| Cold prefill | 4,527 | 4,419 |
| 100k-context decode | 176.8 tok/s (69.1 steps/s) | 66.8 steps/s (tok/s not re-measured) |
| Five temperature-0 answers | reference | 4 of 5 differ in wording |

It loads and serves. The pool is the gain. Decode is slower at every user count
because these cards have no FP8 units and the attention kernel pays to unpack
the cache every step; the answers drift as expected from a coarser cache.
halt95's calibrated FP8 KV scales are for their engine and are not used here.
BF16 stays the default; FP8 is for a user who needs the pool more than the
speed.

### Long context

`kv_dtype_check.py` (a 100,057-token prompt, three runs, 400 tokens pinned):
time to first token 21.9 s (4,570 tok/s prefill), then **176.8 tok/s** of
decode (183.9 / 176.8 / 175.2). About 15 % below the short-context rate.

### Draft count

Measured on a second engine with the same checkpoint (vLLM 0.29.1 with the
`ple1` patches and a graph-compatible table path; raw
`cmp-recipe-bench-merlin-oct14-mtp{1,2,3}-1018.*`, `conc-steps-merlin-oct14-mtp*`,
`mtp-sweep-1018.log`), so the absolute numbers are that engine's, but the
trend is what set this recipe's default of 3:

| drafts | 1 user, structured / code / prose | 8 users, total | step | pool at 131k |
|---|---:|---:|---:|---:|
| 1 | 153.1 / 155.9 / 141.2 | 748.5 / 711.7 / 651.8 | 12.6 ms | 1,474,560 |
| 2 | 182.2 / 197.5 / 166.4 | 784.6 / 795.0 / 680.0 | 15.0 ms | 1,412,664 |
| 3 | 184.7 / 217.0 / 167.5 | 782.0 / 864.8 / 669.3 | 16.6 ms | 1,357,804 |

Each draft adds about 2 ms to the step; the second is accepted 60 to 83 % of
the time and pays everywhere, the third 37 to 70 % and pays on code. A fourth
would be accepted too rarely to cover its cost and was not run.

### Things that did not help, so you need not try them

All on the second engine above unless noted (raw files in the same folder):

- **Sharding experts across the two cards** (`--enable-expert-parallel` off):
  refused by the engine — an expert's inner width is 640, the INT4 scale groups
  are 128 wide, and 640 / 2 is not a multiple of 128. Whole experts per card is
  the only TP2 layout for this checkpoint.
- **Locking the GPU clocks**: no change. During decode with three drafts the
  cards draw 176 W, at the 180 W cap, and sit at 1400 to 1499 MHz because of
  it; a lock cannot raise a power-bound clock.
- **`NCCL_PROTO=LL`**: one user level, eight users −5 to −6 %, cold prefill
  −29 %. The TP all-reduce already uses vLLM's custom kernel; NCCL only carries
  the expert exchange, and the low-latency protocol is wrong for prefill's
  large messages.
- **A smaller prefill slice for fairness** (`MAX_BATCHED` 512 on the second
  engine): a decoding user kept 24 tok/s through another request's 30k prompt
  instead of 7.5, for −4 % cold prefill. On Morrowmake's engine the proper
  lever is `--prefill-chunk-with-decodes`, which has not been measured on this
  build yet.

## Notes on the method

- The decode and prefill tables follow MiaAI-Lab's protocol as Morrowmake's
  recipes use it, with MiaAI-Lab's category prompts as data: replies pinned to
  exactly 400 tokens (`min_tokens` + `ignore_eos`), temperature 0, thinking off,
  a salt per request so the prefix cache stays out; decode rate =
  (completion tokens − 1) / (last chunk − first content token). Multi-user rows
  are aggregate tokens over the batch's wall time, prefill and client overhead
  included. All rates are up-to figures from one box.
- Step times come from the server's `vllm:iteration_tokens_total_count`
  counter, one per engine step, not from the drafts counter (which is per
  request per step and halves the apparent step time at two users).
- `kv_dtype_check.py` counts tokens from the server's `usage`, not streamed
  chunks: with MTP a chunk carries every accepted token of a step, and a first
  version of the script that counted chunks reported the step rate as tok/s.
