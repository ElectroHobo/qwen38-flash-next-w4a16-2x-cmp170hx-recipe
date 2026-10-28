# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%2018305948ac-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8)
![release](https://img.shields.io/badge/release-0.3.0-2ea44f)

**Qwen3.8-Flash-Next (125B MoE) on two CMP 170HX cards with the full
262,144-token context: 241 / 350 / 198 tok/s for one user (structured / code /
prose), 495 tok/s editing code, 895 / 987 / 770 tok/s across eight, and up to
128 requests at once — 1,937 tok/s in total with the default drafter, 1,966 with
the MTP mode for many users. INT4 experts, the n-gram table in host RAM or in a file on disk (one copy
can serve several servers), the whole decode step in one CUDA graph, inco.ai's
DFlash 2 drafter, RecoverSSM for the linear-attention layers, bf16 KV cache and
recurrent states — and an optional host-RAM tier that brings an evicted
100k-token context back in under a second instead of re-reading it in 21 s.**

**One card instead of two?** The sibling recipe
[qwen38-flash-next-3bpw-1x-cmp170hx-recipe](https://github.com/ElectroHobo/qwen38-flash-next-3bpw-1x-cmp170hx-recipe)
serves a 3 bpw build of the same model on a single CMP 170HX with the same engine,
patches and n-gram table, so both can run on one host from one copy of the table.

**New in 0.3.0: the n-gram table on disk, for hosts with less RAM.** Set
`PLE_SHARED_DIR` to a folder on a disk filesystem instead of tmpfs and the 48 GiB
table is a file there: the cards read it through HMM, still inside the one CUDA
graph per step, and the page cache keeps in RAM as much of it as the host can spare.
Without that setting 0.3.0 runs as 0.2.0. From fresh clones, 2026-10-27:

| | table in RAM (default) | table on disk |
|---|---:|---:|
| 1 user, structured / code / prose | 234.9 / 358.6 / 200.8 | 232.8 / 344.4 / 199.3 |
| 8 users, total | 885.9 / 980.9 / 753.0 | 878.8 / 976.0 / 726.2 |
| 128 users, total | 1,934 | 1,851 |
| Cold prefill | 4,723 | 4,000 |
| HumanEval / GSM8K (first 200) | 162/164, 198/200 (0.2.0) | 162/164, 196/200 |
| Host memory the server holds | the 48 GiB table pinned (64 GiB as allocated) + 14 GiB | 14 GiB, plus what the page cache keeps of the table |
| Whole container capped at 22 GiB, 8 users | does not fit | 840.8 / 946.1 / 624.9 |

It needs NVIDIA's open kernel modules with HMM (`nvidia-smi -q` shows
`Addressing Mode: HMM`; `./start.sh` checks) and 48 GiB of disk for the file.
How to use it: [The table on disk](docs/how-to-use.md#the-table-on-disk); what it
costs and why: [results](docs/results.md#release-030).

**New in 0.2.0: many users.** For one to eight users 0.2.0 is as fast as 0.1.0
(the differences are within run-to-run spread); what changes is how many fit at
once and the total they get:

| | 0.1.0 | 0.2.0 |
|---|---:|---:|
| Requests at once | 8 | **128** |
| Most total tok/s, same box, mixed-load development script | 609 (8 users, its limit) | **1,555** DFlash 2, **1,814** MTP (128 users) |
| 1 user, structured / code / prose | 253.2 / 348.9 / 203.2 | 241.2 / 349.7 / 198.4 |
| 8 users, total | 842.2 / 992.1 / 726.7 | 894.5 / 986.8 / 769.7 |
| Cold prefill | 4,729 | 4,696 |
| KV pool | 1,398,101 tokens | 1,475,836 tokens |
| HumanEval / GSM8K (first 200) | 163/164, 197/200 | 162/164, 198/200 |

The 1-to-8-user rows are each release's fresh-clone run (`cmp_recipe_bench.py`);
on the release sweep 0.2.0 does 1,937 tok/s at 128 users with DFlash 2 and 1,966
with MTP ([Many users](#many-users-recoverssm-bf16-states-and-which-drafter)).
Also new: one n-gram table for several servers on a host (`PLE_SHARED_DIR`), and
a fix for an out-of-memory under bursts of shared prompts.

Serve [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
on two NVIDIA CMP 170HX cards with [Morrowmake's vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
fork of vLLM: an OpenAI-compatible API with the model's full context, DFlash 2
speculative decoding, vision, reasoning and tool calls. The engine runs
only in a container pinned by digest. One command from a fresh clone, after
the [one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe
cd qwen38-flash-next-w4a16-2x-cmp170hx-recipe
./start.sh          # pull the engine image, fetch the model, start, wait for /health
```

`./start.sh` skips anything already done, so running it twice is safe; an
interrupted download resumes. The checkpoint is downloaded at the exact
Hugging Face revision this release was validated with.

**Prefer Docker Compose?** The same container, also one command from a fresh
clone: `docker compose up` (or `docker compose up -d`). Use one or the other,
not both.

> **This is a community recipe.** It is made and maintained by its author, not
> by Morrowmake, who has not tested, endorsed or supported it, and not by
> halt95, whose checkpoint it serves on an engine they did not build it for.
> It runs on Morrowmake's engine and image and descends from their
> [Qwen3.8-Flash-Next EXL3 recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe);
> what is ours is the checkpoint choice, three configuration changes, the
> engine patches in [`patches/`](patches/) mounted over the image's (DFlash 2
> support, RecoverSSM, the shared n-gram table, a copy-on-write fix), the KV
> offload wiring, the benchmarks and the measurements. **Contact for issues:
> @ElectroHobo, in
> [this repository's issues](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe/issues)** —
> not Morrowmake's. Made to the shape of Morrowmake's
> [community recipes guidelines](https://github.com/Morrowmake/vllm-cmp170hx/blob/ampere/COMMUNITY_RECIPES.md),
> which say the same about support and endorsement.

## Why this build exists: old CPUs

This recipe was made for a server with **2016-era CPUs** (dual Xeon E5-2697A v4,
Broadwell, 3.5 GHz) and a lot of RAM.

Morrowmake's EXL3 recipe reads the model's 51B-parameter n-gram table from the
SSD in the middle of every decode step, and that read breaks the step into
pieces the CPU has to drive in Python. On a current desktop CPU that costs
little; on a Broadwell it costs **about 44 ms per step**, the GPUs idle for
most of it, and the recipe runs at 55 to 70 tok/s for one user against its
published 153. The weight format has nothing to do with it: we measured a
W4A16 checkpoint on the same engine in the same piecewise mode at the same
44 ms.

This build keeps the table in host RAM instead (it is FP8 in halt95's
checkpoint, 51 GB), which lets the engine capture **the whole decode step as
one CUDA graph**. The step drops to 15 ms and the CPU stops mattering: on the
same Broadwell box this recipe is faster for one user than the EXL3 recipe is
on Morrowmake's hardware. If your host has a current CPU, the EXL3 recipe may
suit you as well or better (its weights are 5.05 bpw, a finer quantisation
than INT4, and it needs no RAM for the table); if your host is old and has
96 GB or more of RAM, this is the one.

Three lines separate this recipe's `serve.sh` from Morrowmake's: the checkpoint
loads through vLLM's ordinary loader (`--load-format auto`), the table is
offloaded to host RAM (`cpu_offload: true`), and full CUDA graphs are allowed
and made the default. Plus the KV offload block, carried over from our
[MiMo recipe](https://github.com/ElectroHobo/mimo-v26-flash-cmp170hx-recipe)
on the same engine, the DFlash 2 drafter and the many-users changes below.
Everything else is theirs.

### The drafter: DFlash 2

The checkpoint carries its own MTP draft head, and that is what first ran
here. [inco.ai's DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
for this model drafts a whole block in one pass instead of running a head once
per draft, so more drafts cost little. Morrowmake's engine already has the
DFlash 2 speculator; the model code did not let a drafter attach, and two
engine files in [`patches/`](patches/) add that (48 lines; mounted over the
image's, which is otherwise used as released). With seven drafts for one or two
requests, five for three or four, three to 32 and two beyond, it is **44 %
faster than MTP on code for one user and 83 % faster editing code**, with the
same quality. The port and every measurement are in
[docs/results.md](docs/results.md), "DFlash 2". `SPEC_MODE=mtp` runs the MTP
head instead.

### Many users: RecoverSSM, BF16 states, and which drafter

New in 0.2.0. The model's linear-attention (Gated DeltaNet) layers keep a
recurrent state per request, and to verify drafts the engine kept one state per
draft position: with seven drafts every request reserved eight state pages, and
at 48 requests allowed only 28 fitted. **RecoverSSM** (an engine patch in
[`patches/`](patches/), the design Morrowmake's engine uses for GLM) keeps one
state per request and replays the accepted tokens after sampling, so **128
requests fit** with room to spare; the states are kept in BF16 (half the pages,
steps 1–3 % shorter at load). Quality is level ([docs/results.md](docs/results.md),
"Release 0.2.0").

Which drafter to run depends on the load. DFlash 2 (the default) is the faster
one up to about sixteen users, and by far on code; the checkpoint's MTP head with
a 3 / 2 / 1 draft schedule (`SPEC_MODE=mtp`) is ahead from about 32 users up. Measured from a fresh clone, total tok/s, 700-token replies at
temperature 0 (`bench/concurrency_sweep.py`):

| Users | 1 | 4 | 8 | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash 2 (default) | **216** | **503** | **772** | **1,067** | 1,357 | 1,484 | 1,647 | 1,762 | 1,937 |
| MTP (`SPEC_MODE=mtp`) | 198 | 496 | 744 | 1,041 | **1,379** | **1,543** | **1,732** | **1,903** | **1,966** |

On a mixed workload (requests of different lengths and topics at once) MTP's lead
from 16 users up was larger in our runs, 7 to 17 %. Per user, 128 at once get
about 15 tok/s each; lower `MAX_SEQS` if each user's speed matters more than the
total. One server cannot switch drafters by load. To have both, run a second
server with the other drafter: it needs its own two cards and about 16 GB of host
RAM, but with the shared n-gram table not a second copy of the 48 GiB table
([docs/how-to-use.md](docs/how-to-use.md), "Several servers, one table").

## One-time host setup

- **2× NVIDIA CMP 170HX, each exposing 64 GiB**; PCIe x16 links (the x16
  capacitor modification). Peer-to-peer verified at start.
- **96 GB of host RAM** with the n-gram table in RAM (it takes 48 GiB, 64 GiB
  as the engine allocates it; the engine and the host need the rest), **plus
  `KV_OFFLOAD_GB`** if you turn the KV tier on. **Much less with the table on
  disk** (`PLE_SHARED_DIR` on a disk filesystem, 0.3.0): the server holds about
  14 GiB and the page cache keeps the rows in use; capped at 22 GiB in all it
  ran within 3–4 % of uncapped on structured and code and 14 % on prose, on our prompts
  ([results](docs/results.md#how-much-ram-the-table-on-disk-needs)). Measured on
  204 GiB (0.1.0, 0.2.0) and 352 GB (0.3.0).
- Linux with NVIDIA driver **580 or newer** (measured on 610.57.04); for the
  table on disk, the open kernel modules with HMM addressing.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  usable by your user: `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists the cards.
- Optional: a P2P-capable driver build for verified peer-to-peer; without it
  the server runs with P2P disabled (`P2P=auto` checks at every start).
- For `./start.sh`: `git`, `curl`, `flock`, `setsid`, Python 3 for the boot
  check, and `jq` for the smoke test.
- About 130 GiB of free disk for the checkpoint (115.3 GiB), plus the engine
  image and the kernel compile caches. The disk only matters while loading.

## Hardware

| | |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX (sm_80), 65,536 MiB each, power limit 180 W |
| Links | PCIe Gen2 x16 per card; peer-to-peer verified at start |
| Host | 2× Xeon E5-2697A v4 (Broadwell, 16 cores each), 204 GiB RAM |
| Driver | 610.57.04 (580 or newer) |
| Disk | ~130 GiB free for the checkpoint; any disk, it is read once per start |
| Software | Docker with the NVIDIA Container Toolkit |

Every weight sits on the two GPUs except the n-gram (PLE) table, which the
engine copies into host RAM at start. Two cards only: tensor-parallel 2 with
whole routed experts on each card (the INT4 scale groups are 128 wide and an
expert's inner width is 640, so an expert cannot be split across two cards).
Pick the cards with `CUDA_VISIBLE_DEVICES` (two distinct nvidia-smi numbers).

## Engine

| | |
|---|---|
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8) at `18305948acba3f4c4d912cdd56f2202d2ebd94e8` |
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`, Morrowmake's release image for their Qwen EXL3 recipe, used as released apart from the mounted files. The first start compiles kernels and RecoverSSM's verify (about 22 minutes), later starts take 5 to 11 |
| Checkpoint | `halt95/Qwen3.8-Flash-Next-W4A16-Merlin`, revision `cec53e9f90e5995b81691604ec0b32191debf797` (weights unchanged since 2026-09-08 per the card) |
| Precision | routed experts INT4 g128 (Intel AutoRound), Gated DeltaNet projections INT8, attention / norms / embeddings / lm_head / shared expert / vision BF16, n-gram table FP8 in host RAM, **KV cache BF16**, indexer cache BF16, **GDN recurrent states BF16** (the checkpoint asks for fp32; `SSM_DTYPE=float32` restores it) |
| Drafter | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6` (Apache-2.0, 1.1 GiB): 7 drafts for one or two requests, 5 for three or four, 3 to 32, 2 beyond. Alternative: the checkpoint's MTP head (`SPEC_MODE=mtp`), 3 drafts to four requests, 2 to eight, 1 beyond |
| Engine patches | files in [`patches/`](patches/), mounted read-only over the image's: the DFlash 2 capture hook and speculation-method check (dflash mode), RecoverSSM for the GDN layers (a new package, its CUDA verify built at first start, and three edited files), the shared n-gram table loader and chunked copy-on-write (every mode) |
| Requests | up to 128 at once (`MAX_SEQS`), 94 % of the KV pool with 128 running |
| CUDA graphs | `FULL_AND_PIECEWISE`: the decode step captured whole |

halt95 built this checkpoint for four RTX 3090s on their own vLLM patches, with
an FP8 KV cache and calibrated scales. Here it runs on Morrowmake's engine
through vLLM's compressed-tensors path; the FP8 KV scales are not used (see
[docs/results.md](docs/results.md) for the FP8 KV measurement).

### Settings, exactly

What the container runs with the defaults (`DRY=1 ./start.sh` prints it):
`vllm serve /models/Qwen3.8-Flash-Next-W4A16-Merlin --tensor-parallel-size 2
--pipeline-parallel-size 1 --enable-expert-parallel --dtype bfloat16
--kv-cache-dtype bfloat16 --load-format auto --mm-encoder-tp-mode data
--attention-config '{"indexer_kv_dtype":"bf16"}' --engram-config '{"cpu_offload":true}'
--max-model-len 262144 --gpu-memory-utilization 0.95 --max-num-seqs 128
--max-num-batched-tokens 3460 --long-prefill-token-threshold 0
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
--prefix-match-unit 16 --block-size 912 --mamba-ssm-cache-dtype bfloat16
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
--speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,2,7],[3,4,5],[5,32,3],[33,128,2]]}'`,
with `KV_OFFLOAD_GB` adding `--kv-transfer-config` for the `OffloadingConnector`
(`TieringOffloadingSpec`) and `"disable_eagle_block_drop":true` to the
speculative config, which offload needs with a drafter (vLLM #52735). The
drafter is mounted at `/drafter`, the files from `patches/` over their engine
paths ([`patches/README.md`](patches/README.md)). Environment:
`VLLM_GDN_RECOVER=1` (RecoverSSM), `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
`VLLM_QWEN4_QSA_TOPK_CANONICAL=1`, `VLLM_CUSTOM_ALLREDUCE_FLAGS=1`,
`VLLM_THIN_GEMM=1`, `VLLM_THIN_GEMM_GLUON=1`, `VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, and after the P2P check
passes `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1` with `NCCL_P2P_LEVEL=SYS`;
`CUDA_DEVICE_ORDER=PCI_BUS_ID`; with `PLE_SHARED_DIR` also
`VLLM_QWEN_PLE_SHARED_DIR`, `VLLM_QWEN_PLE_SHARED_NAME` and
`VLLM_QWEN_PLE_SHARED_PIN` (`1` on tmpfs, `0` on disk). Against Morrowmake's
Qwen recipe the differences are `--load-format auto` (theirs: `exl3`),
`cpu_offload: true` (theirs: `false`), `FULL_AND_PIECEWISE` (theirs:
`PIECEWISE`), the DFlash 2 drafter with its schedule, block size 912 and
prefix-match unit 16 (theirs: MTP, 3 drafts, 64), 128 requests (theirs: 8),
BF16 GDN states and RecoverSSM, and the optional KV offload; everything else is
their default.

## Headline numbers

Measured 2026-10-23 from a fresh clone of this repository (empty `cache/`, bare
`./start.sh`) on 2× CMP 170HX, 74 SMs per card, P2P on, recipe defaults (no KV
tier), 262,144-token context, 180 W per card and PCIe x16 links, under the
protocol Morrowmake's recipes use (MiaAI-Lab's: 400 tokens, temperature 0,
thinking off; one user is the median of five streaming runs, two to eight users
the median aggregate of three runs). Raw files:
[`bench/results/release-0.2.0/`](bench/results/release-0.2.0/).

| | This recipe, DFlash 2 (default) | This recipe, MTP (`SPEC_MODE=mtp`) | Morrowmake's EXL3 recipe, their published figures |
|---|---:|---:|---:|
| **1 user** · structured / code / prose | **241.2 / 349.7 / 198.4** | 208.1 / 242.9 / 184.1 | 152.8 / 152.6 / 118.3 |
| **2 users, total** | 374.4 / 489.3 / 288.4 | 330.4 / 387.7 / 299.0 | 236.5 / 265.0 / 204.6 |
| **4 users, total** | 556.2 / 730.8 / 489.1 | 538.2 / 569.0 / 476.5 | 357.1 / 458.6 / 327.2 |
| **8 users, total** | **894.5 / 986.8 / 769.7** | 841.7 / 829.3 / 728.4 | 588.2 / 674.7 / 489.9 |
| **128 users, total** (`concurrency_sweep.py`, 700 tokens) | 1,937 | **1,966** | — |
| **Prompt reading** (cold prefill) | **4,696** | 4,520 | 2,651 |
| **KV pool** (tokens) | **1,475,836** | 1,438,968 | 1,044,325 |
| Code editing, one user (`bench/agent_probes.py edit`) | **495** | 271 | — |

Their column is from their README, measured on their box with a current CPU;
the EXL3 recipe on *our* box runs at 55 to 70 tok/s for one user. Different
weights too: EXL3 at 5.05 bpw against INT4 experts here, and a different
drafter. Against our 0.1.0 (DFlash 2, up to 8 requests: 253 / 349 / 203 for one
user, 842 / 992 / 727 at eight) one to eight users are level within run-to-run
spread; what 0.2.0 adds is everything above eight.

### Other two-card recipes

Two other recipes publish the same protocol on 2× CMP 170HX, so they compare
directly. Their numbers are from their READMEs on 2026-10-19; nothing here was
re-run by us.

| | This recipe (Morrowmake's engine, TP2, x16 links, Broadwell host) | [AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next) (own vLLM 0.30.0 fork, PP2, Intel W4A16, table on NVMe, x4 links, Core Ultra 5 host) | [mannydelrio](https://github.com/mannydelrio/qwen38-flash-next-2x-cmp170hx) (vLLM, AWQ g32 uncensored, 4-bit table on SSD, x4 links, Ryzen 5 host, 16 GB RAM) |
|---|---:|---:|---:|
| 1 user · structured / code / prose | 241.2 / **349.7** / **198.4** | **250.8** / 226.2 / 157.4 | 192.4 / 177.8 / 145.4 |
| 2 users, total | 374.4 / **489.3** / **288.4** | **406.1** / 362.9 / 212.2 | 324.5 / 242.8 / 238.9 |
| 4 users, total | 556.2 / **730.8** / **489.1** | **697.4** / 560.2 / 384.8 | 598.2 / 446.2 / 399.2 |
| 8 users, total | 894.5 / **986.8** / **769.7** | **999.5** / 840.4 / 566.8 | 986.4 / 734.2 / 619.6 |
| Cold prefill | 4,696 | **6,300–7,100** (4k–64k prompts) | 7,006 |

So this recipe leads on code and prose at every user count, and AlexBrandes on
structured output; both of the others read prompts 33 to 50 % faster. Neither
publishes more than eight users; this recipe serves up to 128. Different
engines, checkpoints and link widths; the one thing this recipe has that they
do not is the host-RAM table with the whole step in one CUDA graph, which is
what makes it fast on an old CPU, the DFlash 2 drafter and the KV offload tier. If prefill speed or
structured output is what you serve, read theirs too.

| Other measurements | Value |
|---|---|
| Quality (thinking on, `reasoning_effort` medium, temperature 0.6, 16,000-token budget) | **HumanEval 162/164, GSM8K 198/200** (first 200 problems) with DFlash 2, 162/164 and 197/200 with MTP, from the fresh clone; 0.1.0 scored 163/164 and 197/200 |
| KV offload, 64 GiB RAM tier (0.1.0) | a 100k-token context evicted by 1.8M tokens of other traffic comes back in **0.89 s** (21–22 s without the tier); content checked correct |
| Follow-up turn | the second question in an 18.7k-token conversation starts in 0.42 s with DFlash 2, 0.25 s with MTP (the first in 3.9–4.1 s) |
| Two servers, one table | this recipe and the one-card one on one host: 49 GiB of host shared memory together, against 64 GiB for either alone without `PLE_SHARED_DIR`; one user here 241.0 / 347.0 / 196.0 with both up |
| KV offload, NVMe tier behind a 16 GiB RAM tier (0.1.0) | the same reload from disk in **1.85 s** |
| 100k-token context (0.1.0) | prompt read in 21.9 s; decode after it 177 tok/s |
| FP8 KV cache (not the default; 0.1.0) | pool 2,578,229 tokens (+73 %), one-user decode −8 to −10 %, answers shift |

## Using it

`./start.sh` runs preflight (read-only GPU, link, disk and port checks),
installs the engine, downloads the checkpoint, launches the server detached
(output in `logs/serve.log`) and waits for `/health`, then runs a two-request
startup check. Commands:

| Command | What it does |
|---|---|
| `./start.sh` | preflight, install, download, launch (default) |
| `./start.sh install` | the engine image only |
| `./start.sh download` | the checkpoint only |
| `./start.sh stop` / `restart` | stop or restart the server this checkout started |
| `./start.sh status` | server, `/health`, KV cache line |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh update` | `git pull`, pull the image if the pin moved, restart |
| `./start.sh smoke` | one chat request and one tool call |
| `DRY=1 ./start.sh` | print what would happen, change nothing |

The API listens on `127.0.0.1:8000` as model `qwen3.8-flash-next`, with no API
key. Reasoning comes back in the message's `reasoning` field and the answer in
`content`; size `max_tokens` for both; `chat_template_kwargs: {"enable_thinking": false}`
turns reasoning off. Examples: [`examples/`](examples/). Docker Compose:
[`docker-compose.yml`](docker-compose.yml). Step by step, serving other
machines, the KV tier: [docs/how-to-use.md](docs/how-to-use.md).

## Options

Settings go in `.env` (copied from [`.env.example`](.env.example) on the first
run); a prefix assignment beats `.env` for one run
(`CUDA_VISIBLE_DEVICES=2,3 ./start.sh restart`). Every option:

| Setting | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | the two cards (exactly two, distinct) |
| `HOST` | `127.0.0.1` | listen address; `0.0.0.0` for every interface (set `API_KEY` first) |
| `PORT` | `8000` | API port |
| `API_KEY` | unset | require `Authorization: Bearer <key>` |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | model id clients send |
| `MODELS_DIR` | `./models` | checkpoint location (any disk) |
| `KV_OFFLOAD_GB` | `0` | GiB of host RAM for KV blocks evicted from the cards; `0` is off. 64 held ~2.2 M tokens (~30 KB per token) |
| `KV_OFFLOAD_DIR` | unset | a disk tier behind the RAM tier (needs `KV_OFFLOAD_GB`); write-through, never pruned — give it a volume of its own |
| `GRAPH_MODE` | `FULL_AND_PIECEWISE` | `PIECEWISE` is the EXL3 recipe's mode and is ~3× slower per step on an old CPU |
| `P2P` | `auto` | `auto` (content-checked), `off`, `force` |
| `SPEC_MODE` | `dflash` | `dflash` (inco.ai's DFlash 2 drafter; the faster for up to about 16 users), `mtp` (the checkpoint's MTP head; the faster from about 32) or `none` (no speculation; set `BOOT_CHECK=0`, the startup check expects drafting) |
| `SPEC_N` | `7` (dflash) / `3` (mtp) | draft tokens per step; at most 7 for dflash |
| `DRAFTER_REPO` / `DRAFTER_REVISION` | inco.ai's, pinned | the DFlash 2 drafter to download, next to the checkpoint in `MODELS_DIR` |
| `LOCAL_ARGMAX` | `0` | `1`: per-card draft argmax, same drafts |
| `SPEC_SCHEDULE` | `[[1,2,7],[3,4,5],[5,32,3],[33,128,2]]` (dflash) / `[[1,4,3],[5,8,2],[9,128,1]]` (mtp) | draft count by number of requests; `SPEC_SCHEDULE=` for `SPEC_N` drafts at every count |
| `MAX_SEQS` | `128` | concurrent sequences; 128 use 94 % of the KV pool, fewer keep each user faster |
| `RECOVER_SSM` | `1` | RecoverSSM for the GDN layers; `0` = the engine's per-draft states (about 28 requests fit with 7 drafts) |
| `SSM_DTYPE` | `bfloat16` | GDN recurrent state precision; `float32` is the checkpoint's |
| `BLOCK_SIZE` | `912` (`1680` with `float32` states) | attention block size in dflash mode: a multiple of 48 at least the GDN state page |
| `PLE_SHARED_DIR` | unset | where the n-gram table lives, shared by every server on this host given the same folder (this recipe's and the one-card recipe's): on tmpfs (e.g. `/dev/shm/qwen38-ple`) one 48 GiB copy in RAM; on a disk filesystem (e.g. `/srv/qwen38-ple`) a 48 GiB file read through HMM, for hosts with less RAM ([how-to](docs/how-to-use.md#the-table-on-disk)) |
| `PLE_SHARED_NAME` | `qwen38-flash-next-ple-fp8` | the table's file name there; servers that share use the same name |
| `MAX_BATCHED` | `3460` | tokens per scheduler step |
| `REASONING_PARSER` / `TOOL_PARSER` | `qwen3` / `qwen3_coder` | output parsers |
| `EXTRA_ARGS` | empty | extra vLLM arguments; options that change the deployment are refused |
| `BOOT_CHECK` | `1` | two-request startup check after `/health` |
| `READY_TIMEOUT` / `STOP_TIMEOUT` | `1800` / `120` | seconds |

The context (262,144) and GPU memory fraction (0.95) are fixed: `serve.sh`
refuses other values. `FIRST_LIGHT=1` is a 32K/eager/no-MTP diagnostic, not a
serving mode. `PLE_PLACEMENT` and the `VLLM_EXL3_*` switches are the EXL3
reader's and do nothing for this checkpoint; the engine switches that still
apply (`VLLM_QWEN4_QSA_TOPK_CANONICAL`, `VLLM_CUSTOM_ALLREDUCE_FLAGS`,
`VLLM_THIN_GEMM`, `VLLM_THIN_GEMM_GLUON`, `VLLM_FLA_AUTOTUNE_SKIP_SPILLING`,
`VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`) are on by default with a one-variable kill
switch each, as in Morrowmake's recipe. Path and preflight settings are in
[`.env.advanced.example`](.env.advanced.example).

## Update

- `./start.sh update` pulls the new release and engine image and restarts; your
  `.env` is preserved.
- With Compose: `git pull`, then `docker compose up -d`.
- Older releases are git tags; their images may be retired.

## Licence

The recipe is MIT ([`LICENSE`](LICENSE)); the scripts descend from
Morrowmake's MIT-licensed Qwen3.8-Flash-Next recipe. The engine is Apache-2.0.
The checkpoint is under the Qwen Community License 1.0 and carries NVIDIA and
Intel components with their own terms: [docs/model-licences.md](docs/model-licences.md).
The benchmark prompts come from MiaAI-Lab's repository, used as data with
attribution.

## Credits

- **Morrowmake** — the [vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
  engine and image this runs on, and the
  [Qwen3.8-Flash-Next recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
  (MIT) whose scripts (`start.sh`, `serve.sh`, the P2P and boot checks, Compose,
  the examples, the image build) this recipe descends from. Not involved in
  this recipe.
- **halt95** — the [W4A16 checkpoint](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
  and its [companion engine](https://github.com/halt95/qwen38-flash-next-3090s)
  for RTX 3090s, whose measurements and card pointed the way. Not involved in
  this recipe, which runs their weights on an engine they did not build it for.
- **Intel** — the [AutoRound INT4 experts](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
  inside the checkpoint; **RadixArk** — the [FP8 n-gram table](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4);
  **DominikBucko** — the MTP-head packing recipe. All three credited by halt95.
- **Qwen** — [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  (Qwen Community License 1.0).
- **vLLM** — the [project](https://github.com/vllm-project/vllm) (Apache-2.0)
  Morrowmake's engine forks, including the Qwen3.8-Flash-Next model code, the
  n-gram table offload to host RAM (Engram `cpu_offload`) and the KV offloading
  connector this recipe turns on.
- **MiaAI-Lab** — the benchmark protocol and category prompts (from their DGX
  Spark kits, AGPL-3.0; the prompts are used here as data, the harness is our own).
- **inco.ai** — the [DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  for this model (Apache-2.0) that this recipe drafts with, and the capture
  patch shipped with it that says what the drafter was trained on.

If this recipe helps you, credit the people above and link their work.
Citation metadata for this repository: [`CITATION.cff`](CITATION.cff).
