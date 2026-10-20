# Results

Every number here was measured on the same box: 2× CMP 170HX (74 SMs, 64 GiB,
180 W, PCIe Gen2 x16, P2P on) in a dual Xeon E5-2697A v4 (Broadwell) host with
204 GiB of RAM, driver 610.57.04, Morrowmake's engine at
`18305948acba3f4c4d912cdd56f2202d2ebd94e8`. Raw files for each run are in
[`bench/results/release-0.1.0/`](../bench/results/release-0.1.0/); the harnesses
are in [`bench/`](../bench/). Dates are 2026-10-18 and 2026-10-19.

## Release 0.1.0

### The serving configuration

Recipe defaults, started from a fresh clone of this repository with `./start.sh`
and `KV_OFFLOAD_GB=64` (startup check passed: 666 of 714 drafts accepted):
`halt95/Qwen3.8-Flash-Next-W4A16-Merlin`, TP2 with whole experts per card, the
n-gram table in host RAM, `FULL_AND_PIECEWISE` CUDA graphs, inco.ai's DFlash 2
drafter scheduled 7 / 5 / 3 by requests, block size 1680, bf16 KV, 262,144
context, `MAX_BATCHED` 3460. `cmp_recipe_bench.py` (MiaAI-Lab's protocol: 400
tokens, temperature 0, thinking off; one user the median of five streaming runs,
more users the median aggregate of three runs, prefill the median of three
real-text prompts repeated three times). Raw:
`cmp-recipe-bench-recipe-0.1.0-fresh.*`, `agent-*-recipe-0.1.0-fresh.json`,
`kv-offload-recipe-0.1.0-fresh-ram64.txt`.

| | DFlash 2 (default) | MTP, 3 drafts (`SPEC_MODE=mtp`) |
|---|---:|---:|
| 1 user, structured / code / prose | **253.2 / 348.9 / 203.2** | 206.4 / 245.6 / 186.2 |
| 2 users, total | 381.1 / 483.8 / 298.1 | 327.3 / 361.9 / 277.0 |
| 4 users, total | 515.2 / 688.6 / 455.8 | 506.9 / 541.1 / 452.1 |
| 8 users, total | 842.2 / 992.1 / 726.7 | 820.7 / 826.6 / 692.4 |
| Code editing, one user (`agent_probes.py edit`) | 497 | 275 |
| Follow-up turn, time to first token (`agent_probes.py followup`) | 0.30 s | 0.25 s |
| Cold prefill (21,868 / 31,224 / 34,905-token prompts) | 4,729 | 4,527 |
| KV pool | 1,398,101 tokens | 1,489,261 tokens |

The MTP column is the same build measured earlier the same day (raw
`cmp-recipe-bench-merlin-morrowmake-fullgraph-1019.*`); the rest of this
section and the next are about that configuration, which is what established
the 15 ms step.

The engine step (`vllm_conc_steps.py`, 400 tokens pinned, from the server's own
`iteration_tokens_total_count`; raw `conc-steps-merlin-morrowmake-fullgraph-1019.txt`):
**15.2 ms at one user, 19.6 at two, 23 to 25 at four.** MTP accepts 0.84 /
0.56 / 0.45 of its three drafts by position on structured prompts and 0.92 /
0.84 / 0.79 on code, for 2.9 to 3.5 tokens per request-step.

### DFlash 2: the drafter, and the two engine files it needs

The default drafter is [inco.ai's DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
for this model (Apache-2.0, revision `a22a1744`), a 5-layer block-diffusion
drafter that proposes a whole block of up to 7 tokens in one pass. Morrowmake's
engine already carries the DFlash 2 speculator (their GLM recipe runs one); the
Qwen3.8-Flash-Next model code did not let a drafter attach. Two engine files in
[`patches/`](../patches/), mounted over the image's copies by `start.sh` and
Compose, add that (48 lines added, 1 changed; [`patches/README.md`](../patches/README.md)):

- `qwen4_exp_nvidia_model.py`: the model declares itself a drafter target and
  hands out the hidden state entering layers 6, 16, 26, 36 and 46, completed
  across the hyper-connection streams and averaged over them — what the
  drafter was trained on (inco.ai's capture patch: completed outputs of layers
  5, 15, 25, 35, 45, averaged over the four streams; the engine adds one to
  DFlash layer ids).
- `models_config.py`: `dflash` added to the speculation methods the model's
  config check accepts.

Measured on the same box, each run on a fresh restart, no KV tier (raw
`cmp-recipe-bench-{dflash2-k3,dflash2-k4,dflash2-k4sched,tune-*}-1019.*`,
`conc-steps-*`, `agent-edit-*`):

| | MTP, 3 drafts | DFlash 2, 3 | DFlash 2, 4 | DFlash 2, 7 | **DFlash 2, 7 / 5 / 3 by users (default)** |
|---|---:|---:|---:|---:|---:|
| 1 user, structured / code / prose | 206 / 246 / 186 | 229 / 273 / 199 | 229 / 315 / 202 | 242 / 357 / 205 | **239 / 353 / 204** |
| 2 users, total | 327 / 362 / 277 | 347 / 368 / 314 | 347 / 436 / 318 | 379 / 487 / 294 | **385 / 485 / 309** |
| 4 users, total | 507 / 541 / 452 | 566 / 602 / 491 | 537 / 659 / 463 | 536 / 705 / 431 | **526 / 686 / 456** |
| 8 users, total | 821 / 827 / 692 | 877 / 951 / 737 | 825 / 1,017 / 702 | 733 / 1,000 / 593 | **847 / 986 / 728** |
| code editing (a 2,222-token file, rename, whole file back) | 275 | 302 | 359 | 501 | **496** |
| step at 1 user | 15.2 ms | 14.0 ms | 13.7–15.4 ms | 15.2–15.6 ms | 15.5–15.9 ms |
| KV pool at 262k | 1,489,261 | 1,557,845 | 1,509,680 | 1,403,318 | 1,398,101 |

At three drafts DFlash 2 accepts about as much as MTP; the gain is the cheaper
step, one drafting pass instead of three MTP head runs. More drafts then pay on
code: at seven, code acceptance by position is 0.92 / 0.82 / 0.73 / 0.67 / 0.56
/ 0.47 / 0.34, 5.5 tokens per step. At eight users the step is the GPU's and
seven drafts cost structured and prose, so the default schedule
(`[[1,2,7],[3,4,5],[5,8,3]]`) drafts 7 for one or two requests, 5 for three or
four, 3 beyond.

Five to seven drafts need `--block-size 1680` (set by `serve.sh` in dflash
mode): the sparse-attention ring for 5–7 drafts is 12 tokens and must divide
the attention block size, and the engine's own choice (1,616–1,648) does not;
1,680 is divisible by 12 and by the prefix-match unit 16 that four or more
drafts need. Without it the server refuses to start ("QSA ring capacity 12
must divide the attention block size 1648").

Quality on the default (`bench/quality_eval.py`, settings as in "Quality"):
HumanEval 163/164, GSM8K 197/200, no reply cut off.

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
| 1 user, structured / code / prose | 152.8 / 152.6 / 118.3 | 253.2 / 348.9 / 203.2 |
| 2 users, total | 236.5 / 265.0 / 204.6 | 381.1 / 483.8 / 298.1 |
| 4 users, total | 357.1 / 458.6 / 327.2 | 515.2 / 688.6 / 455.8 |
| 8 users, total | 588.2 / 674.7 / 489.9 | 842.2 / 992.1 / 726.7 |
| Cold prefill | 2,651 | 4,729 |
| KV pool at 262k | 1,044,325 | 1,398,101 |

What the comparison does not say: the EXL3 weights are 5.05 bits per weight, a
finer quantisation than INT4 experts, and the drafters differ (their MTP head,
our DFlash 2). The table in RAM costs 51 GB of
host memory that the EXL3 recipe does not need.

### Against the other two-card recipes

Two more recipes publish the same protocol on 2× CMP 170HX; the README's
"Other two-card recipes" table puts them side by side. In short: this recipe
leads on code and prose at every user count and is level on structured output
for one user; AlexBrandes's pipeline-parallel build on their own vLLM fork
leads on structured output from two users up; both it and mannydelrio's read
prompts 33 to 50 % faster (6,300 to 7,100 tok/s against 4,729). Neither keeps
the table in host RAM, and neither reports a step-time or CPU dependence; both
run on current desktop CPUs. The prefill gap is the clearest thing left to
close here.

### Quality

`bench/quality_eval.py` (our harness: the model writes the function and the
dataset's own tests run against it in a network-less `python:3.12-slim`
container; GSM8K answers are compared numerically with the gold), thinking on,
`reasoning_effort` medium, temperature 0.6 / top-p 0.95 / top-k 20, 16,000-token
budget, eight requests at a time. Raw `dflash2-k7sched-*` (DFlash 2 default),
`dflash2-w4a16-*` (DFlash 2, 4 drafts), `merlin-w4a16-*` (MTP),
`{humaneval,gsm8k}.jsonl` and `.scores.json`.

| | DFlash 2 (default) | DFlash 2, 4 drafts | MTP, 3 drafts |
|---|---:|---:|---:|
| HumanEval (all 164) | **163 / 164** | 164 / 164 | 161 / 164 |
| GSM8K (first 200) | **197 / 200** | 197 / 200 | 199 / 200 |
| replies cut off | 0 | 0 | 0 |
| median completion tokens, HumanEval | 587 | 605 | 606 |
| median wall per problem, HumanEval | 6.0 s | 6.2 s | 7.6 s |

The differences between the drafters are within temperature-0.6 sampling
noise; drafting is lossless (every draft is verified by the target).

Both suites sit near their ceiling for a model of this size, so what they
establish is that the INT4 experts, the table in host RAM and the unused FP8 KV
scales did not damage the answers; they do not measure long-form writing,
tool-call adherence or hard competition code (LiveCodeBench on this harness
scores zero for models of this class and separates nothing).

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

The table above is the MTP build. With the DFlash 2 default and the same 64 GiB
tier, from the fresh clone: cold 20.75 s, warm 0.56 s, **evicted 0.89 s**,
evicted-again 0.57 s, all three codes correct; `external_prefix_cache_hits_total`
+99,120 tokens and 2.85 GB loaded from the tier (raw
`kv-offload-recipe-0.1.0-fresh-ram64.txt`). The tier needs
`"disable_eagle_block_drop": true` with a drafter, which `serve.sh` adds when
`KV_OFFLOAD_GB` is set.

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

### MTP draft count

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
- **Explicit CUDA graph capture sizes** (on this engine; raw
  `cmp-recipe-bench-merlin-morrowmake-sizes-1019.*`): mannydelrio's build gained
  65 % at eight users by capturing graph sizes up to 32 tokens. Here
  `cudagraph_capture_sizes [4, 8, …, 64]` changed nothing — 200.9 / 237.9 / 185.3
  for one user, 811.9 / 834.3 / 680.5 at eight, the same step times — because
  Morrowmake's breakable-graph capture does not bucket by batch size the way
  stock vLLM does.
- **Pipeline parallel 2** (the layout behind AlexBrandes's and inordic's
  prefill): this engine refuses it for this model — "N-gram PLE embedding
  requires pipeline_parallel_size=1". AlexBrandes's fork carries the patches
  that allow it. On this engine it is a port, not a setting.
- **N-gram drafting instead of MTP** (`method: ngram`, 3 drafts, lookup 2 to
  6; raw `cmp-recipe-bench-spec-ngram-1019.*`, `agent-edit-spec-ngram-1019.json`):
  inordic's build gains 15 % on code editing by looking up drafts in the
  context *before* the MTP head, a chain this engine does not offer. N-gram
  alone loses everywhere here: code editing 226 tok/s against MTP's 275,
  generation 78.7 / 77.7 / 68.8 for one user against 206 / 246 / 186.
  Suffix decoding (`method: suffix`) needs the `arctic-inference` package,
  which the image does not carry.
- **N-gram lookup in front of MTP** (upstream vLLM PR #60615, the change
  behind inordic's +15 %, ported onto this engine and bind-mounted over the
  image for the test; raw `cmp-recipe-bench-ngram-lookup*-1019.*`,
  `agent-edit-ngram-lookup*-1019.json`): it loads and serves correctly, and it
  costs 15 % on every single-request workload here — code editing 234 tok/s
  against 275, generation 172 / 205 / 160 against 206 / 246 / 186, with the
  default 12–16-token window and with 3–8; eight users unchanged, as the patch
  promises. The lookup runs on every single-request step whether it matches or
  not, and on a 15 ms step that fixed cost outweighs the matches; on the
  author's 19 ms steps at 52 tok/s it did not. Not included.

### Agent-shaped workloads

`bench/agent_probes.py`, on the serving configuration:

| | |
|---|---:|
| Code editing: a 2,222-token file in the prompt, rename every function, return the whole file (2,227 tokens out) | **275 tok/s** decode, rename correct |
| Follow-up turn: an 18.7k-token document, a question, then a second question | first turn 4.1 s to first token; **second turn 0.25 s** |

Editing decodes faster than generation because MTP already profits from text
it can copy. The follow-up turn is a prefix-cache hit at the recipe's
`--prefix-match-unit 64`; a smaller unit would not improve on 0.25 s.

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
