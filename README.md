# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%2018305948ac-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8)
![release](https://img.shields.io/badge/release-0.1.0-2ea44f)

**Qwen3.8-Flash-Next (125B MoE) on two CMP 170HX cards with the full
262,144-token context: 253 / 349 / 203 tok/s for one user (structured / code /
prose), 497 tok/s editing code, 842 / 992 / 727 tok/s across eight, 4,729 tok/s
cold prefill and a 1,398,101-token KV pool. INT4 experts, the n-gram table in
host RAM, the whole decode step in one CUDA graph, inco.ai's DFlash 2 drafter,
bf16 KV cache — and an optional host-RAM tier that brings an evicted
100k-token context back in under a second instead of re-reading it in 21 s.**

Serve [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
on two NVIDIA CMP 170HX cards with [Morrowmake's vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
fork of vLLM: an OpenAI-compatible API with the model's full context, DFlash 2
speculative decoding, vision, reasoning and tool calls. The engine runs
only in a container pinned by digest. One command from a fresh clone, after
the [one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/ElectroHobo/qwen38-flash-next-w4a16-cmp170hx-recipe
cd qwen38-flash-next-w4a16-cmp170hx-recipe
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
> DFlash 2 support (two engine files in [`patches/`](patches/) mounted over the
> image's), the KV offload wiring, the benchmarks and the measurements. **Contact for issues:
> @ElectroHobo, in
> [this repository's issues](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-cmp170hx-recipe/issues)** —
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
on the same engine, and the DFlash 2 drafter below. Everything else is theirs.

### The drafter: DFlash 2

The checkpoint carries its own MTP draft head, and that is what first ran
here. [inco.ai's DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
for this model drafts a whole block in one pass instead of running a head once
per draft, so more drafts cost little. Morrowmake's engine already has the
DFlash 2 speculator; the model code did not let a drafter attach, and two
engine files in [`patches/`](patches/) add that (48 lines; mounted over the
image's, which is otherwise used as released). With seven drafts for one or two
requests, five for three or four and three beyond, it is **43 % faster than MTP
on code for one user and 80 % faster editing code**, with the same quality. The
port and every measurement are in [docs/results.md](docs/results.md),
"DFlash 2". `SPEC_MODE=mtp` runs the MTP head on the unmodified image.

## One-time host setup

- **2× NVIDIA CMP 170HX, each exposing 64 GiB**; PCIe x16 links (the x16
  capacitor modification). Peer-to-peer verified at start.
- **96 GB of host RAM** (the n-gram table takes 51 GB and lives there; the
  engine and the host need the rest), **plus `KV_OFFLOAD_GB`** if you turn the
  KV tier on. Measured on 204 GiB with a 64 GiB tier.
- Linux with NVIDIA driver **580 or newer** (measured on 610.57.04).
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
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`, Morrowmake's release image for their Qwen EXL3 recipe, used as released apart from the two mounted files. The first start compiles kernels (about 17 minutes), later starts take about 5 |
| Checkpoint | `halt95/Qwen3.8-Flash-Next-W4A16-Merlin`, revision `cec53e9f90e5995b81691604ec0b32191debf797` (weights unchanged since 2026-09-08 per the card) |
| Precision | routed experts INT4 g128 (Intel AutoRound), Gated DeltaNet projections INT8, attention / norms / embeddings / lm_head / shared expert / vision BF16, n-gram table FP8 in host RAM, **KV cache BF16**, indexer cache BF16 |
| Drafter | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6` (Apache-2.0, 1.1 GiB): 7 drafts for one or two requests, 5 for three or four, 3 beyond. Alternative: the checkpoint's MTP head (`SPEC_MODE=mtp`), 3 drafts |
| Engine patches | two files in [`patches/`](patches/), mounted read-only over the image's in dflash mode: the drafter capture hook and the speculation-method check |
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
--max-model-len 262144 --gpu-memory-utilization 0.95 --max-num-seqs 8
--max-num-batched-tokens 3460 --long-prefill-token-threshold 0
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
--prefix-match-unit 16 --block-size 1680
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
--speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,2,7],[3,4,5],[5,8,3]]}'`,
with `KV_OFFLOAD_GB` adding `--kv-transfer-config` for the `OffloadingConnector`
(`TieringOffloadingSpec`) and `"disable_eagle_block_drop":true` to the
speculative config, which offload needs with a drafter (vLLM #52735). The
drafter is mounted at `/drafter` and the two files from `patches/` over
`/opt/vllm-src/vllm/models/qwen4_exp/nvidia/model.py` and
`/opt/vllm-src/vllm/model_executor/models/config.py`. Environment: `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
`VLLM_QWEN4_QSA_TOPK_CANONICAL=1`, `VLLM_CUSTOM_ALLREDUCE_FLAGS=1`,
`VLLM_THIN_GEMM=1`, `VLLM_THIN_GEMM_GLUON=1`, `VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, and after the P2P check
passes `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1` with `NCCL_P2P_LEVEL=SYS`;
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Against Morrowmake's Qwen recipe the differences
are `--load-format auto` (theirs: `exl3`), `cpu_offload: true` (theirs:
`false`), `FULL_AND_PIECEWISE` (theirs: `PIECEWISE`), the DFlash 2 drafter with
its schedule, block size 1680 and prefix-match unit 16 (theirs: MTP, 3 drafts,
64), and the optional KV offload; everything else is their default.

## Headline numbers

Measured 2026-10-19 from a fresh clone of this repository on 2× CMP 170HX,
74 SMs per card, P2P on, recipe defaults with `KV_OFFLOAD_GB=64`,
262,144-token context, 180 W per card and PCIe x16 links, under the protocol
Morrowmake's recipes use (MiaAI-Lab's: 400 tokens, temperature 0, thinking
off; one user is the median of five streaming runs, two to eight users the
median aggregate of three runs). Raw files:
[`bench/results/release-0.1.0/`](bench/results/release-0.1.0/).

| | This recipe | Morrowmake's EXL3 recipe, their published figures |
|---|---:|---:|
| **1 user** · structured / code / prose | **253.2 / 348.9 / 203.2** | 152.8 / 152.6 / 118.3 |
| **2 users, total** | 381.1 / 483.8 / 298.1 | 236.5 / 265.0 / 204.6 |
| **4 users, total** | 515.2 / 688.6 / 455.8 | 357.1 / 458.6 / 327.2 |
| **8 users, total** | **842.2 / 992.1 / 726.7** | 588.2 / 674.7 / 489.9 |
| **Prompt reading** (cold prefill) | **4,729** | 2,651 |
| **KV pool** (tokens) | **1,398,101** | 1,044,325 |
| Code editing, one user (`bench/agent_probes.py edit`) | 497 | — |

Their column is from their README, measured on their box with a current CPU;
the EXL3 recipe on *our* box runs at 55 to 70 tok/s for one user. Different
weights too: EXL3 at 5.05 bpw against INT4 experts here, and a different
drafter.

### Other two-card recipes

Two other recipes publish the same protocol on 2× CMP 170HX, so they compare
directly. Their numbers are from their READMEs on 2026-10-19; nothing here was
re-run by us.

| | This recipe (Morrowmake's engine, TP2, x16 links, Broadwell host) | [AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next) (own vLLM 0.30.0 fork, PP2, Intel W4A16, table on NVMe, x4 links, Core Ultra 5 host) | [mannydelrio](https://github.com/mannydelrio/qwen38-flash-next-2x-cmp170hx) (vLLM, AWQ g32 uncensored, 4-bit table on SSD, x4 links, Ryzen 5 host, 16 GB RAM) |
|---|---:|---:|---:|
| 1 user · structured / code / prose | **253.2** / **348.9** / **203.2** | 250.8 / 226.2 / 157.4 | 192.4 / 177.8 / 145.4 |
| 2 users, total | 381.1 / **483.8** / **298.1** | **406.1** / 362.9 / 212.2 | 324.5 / 242.8 / 238.9 |
| 4 users, total | 515.2 / **688.6** / **455.8** | **697.4** / 560.2 / 384.8 | 598.2 / 446.2 / 399.2 |
| 8 users, total | 842.2 / **992.1** / **726.7** | **999.5** / 840.4 / 566.8 | 986.4 / 734.2 / 619.6 |
| Cold prefill | 4,729 | **6,300–7,100** (4k–64k prompts) | 7,006 |

So this recipe leads on code and prose at every user count and is level on
structured output for one user; AlexBrandes leads on structured output from two
users up, and both of the others read prompts 33 to 50 % faster. Different
engines, checkpoints and link widths; the one thing this recipe has that they
do not is the host-RAM table with the whole step in one CUDA graph, which is
what makes it fast on an old CPU, the DFlash 2 drafter and the KV offload tier. If prefill speed or
structured output is what you serve, read theirs too.

| Other measurements | Value |
|---|---|
| Quality (thinking on, `reasoning_effort` medium, temperature 0.6, 16,000-token budget) | **HumanEval 163/164, GSM8K 197/200** (first 200 problems), no reply cut off |
| KV offload, 64 GiB RAM tier | a 100k-token context evicted by 1.8M tokens of other traffic comes back in **0.89 s** (21–22 s without the tier); content checked correct |
| Follow-up turn | the second question in an 18.7k-token conversation starts in 0.30 s (the first in 4.0 s) |
| KV offload, NVMe tier behind a 16 GiB RAM tier | the same reload from disk in **1.85 s** |
| 100k-token context | prompt read in 21.9 s; decode after it 177 tok/s |
| FP8 KV cache (not the default) | pool 2,578,229 tokens (+73 %), one-user decode −8 to −10 %, answers shift |

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
| `SPEC_MODE` | `dflash` | `dflash` (inco.ai's DFlash 2 drafter, two engine files mounted), `mtp` (the checkpoint's MTP head, unmodified image) or `none` (no speculation; set `BOOT_CHECK=0`, the startup check expects drafting) |
| `SPEC_N` | `7` (dflash) / `3` (mtp) | draft tokens per step; at most 7 for dflash |
| `DRAFTER_REPO` / `DRAFTER_REVISION` | inco.ai's, pinned | the DFlash 2 drafter to download, next to the checkpoint in `MODELS_DIR` |
| `LOCAL_ARGMAX` | `0` | `1`: per-card draft argmax, same drafts |
| `SPEC_SCHEDULE` | `[[1,2,7],[3,4,5],[5,8,3]]` (dflash) / empty (mtp) | draft count by number of requests; `SPEC_SCHEDULE=` for `SPEC_N` drafts at every count |
| `MAX_SEQS` | `8` | concurrent sequences |
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
