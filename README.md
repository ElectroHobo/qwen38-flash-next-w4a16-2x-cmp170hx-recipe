# Qwen3.8-Flash-Next W4A16 on 2× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%2018305948ac-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8)
![release](https://img.shields.io/badge/release-0.1.0-2ea44f)

**Qwen3.8-Flash-Next (125B MoE) on two CMP 170HX cards with the full
262,144-token context: 206 / 246 / 186 tok/s for one user (structured / code /
prose), 821 / 827 / 692 tok/s across eight, 4,527 tok/s cold prefill and a
1,489,261-token KV pool. INT4 experts, the n-gram table in host RAM, the whole
decode step in one CUDA graph, native MTP speculation, bf16 KV cache — and an
optional host-RAM tier that brings an evicted 100k-token context back in 0.7 s
instead of re-reading it in 22 s.**

Serve [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
on two NVIDIA CMP 170HX cards with [Morrowmake's vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
fork of vLLM: an OpenAI-compatible API with the model's full context, native
MTP speculative decoding, vision, reasoning and tool calls. The engine runs
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
> what is ours is the checkpoint choice, three configuration changes, the KV
> offload wiring, the benchmarks and the measurements. Report problems in
> [this repository's issues](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-cmp170hx-recipe/issues).

## Why this build exists: old CPUs

This recipe was made for a server with **2016-era CPUs** (dual Xeon E5-2697A v4,
Broadwell, 3.5 GHz) and a lot of RAM, which is a common shape for a CMP 170HX
box: the cards are cheap, the hosts they land in are often old.

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
on the same engine. Everything else is theirs.

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
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`, Morrowmake's release image for their Qwen EXL3 recipe, used as released. The first start compiles kernels (about 15 minutes), later starts take about 5 |
| Checkpoint | `halt95/Qwen3.8-Flash-Next-W4A16-Merlin`, revision `cec53e9f90e5995b81691604ec0b32191debf797` (weights unchanged since 2026-09-08 per the card) |
| Precision | routed experts INT4 g128 (Intel AutoRound), Gated DeltaNet projections INT8, attention / norms / embeddings / lm_head / shared expert / vision BF16, n-gram table FP8 in host RAM, **KV cache BF16**, indexer cache BF16 |
| Speculation | native MTP head (INT4-packed in the checkpoint), 3 draft tokens |
| CUDA graphs | `FULL_AND_PIECEWISE`: the decode step captured whole |

halt95 built this checkpoint for four RTX 3090s on their own vLLM patches, with
an FP8 KV cache and calibrated scales. Here it runs on Morrowmake's engine
through vLLM's compressed-tensors path; the FP8 KV scales are not used (see
[docs/results.md](docs/results.md) for the FP8 KV measurement).

## Headline numbers

Measured 2026-10-19 on 2× CMP 170HX, 74 SMs per card, P2P on, recipe defaults,
262,144-token context, 180 W per card and PCIe x16 links, under the protocol
Morrowmake's recipes use (MiaAI-Lab's: 400 tokens, temperature 0, thinking
off; one user is the median of five streaming runs, two to eight users the
median aggregate of three runs). Raw files:
[`bench/results/release-0.1.0/`](bench/results/release-0.1.0/).

| | This recipe | Morrowmake's EXL3 recipe, their published figures |
|---|---:|---:|
| **1 user** · structured / code / prose | **206.4 / 245.6 / 186.2** | 152.8 / 152.6 / 118.3 |
| **2 users, total** | 327.3 / 361.9 / 277.0 | 236.5 / 265.0 / 204.6 |
| **4 users, total** | 506.9 / 541.1 / 452.1 | 357.1 / 458.6 / 327.2 |
| **8 users, total** | **820.7 / 826.6 / 692.4** | 588.2 / 674.7 / 489.9 |
| **Prompt reading** (cold prefill) | **4,527** | 2,651 |
| **KV pool** (tokens) | **1,489,261** | 1,044,325 |
| Decode step, one user | 15.2 ms | — |

Their column is from their README, measured on their box with a current CPU;
the EXL3 recipe on *our* box runs at 55 to 70 tok/s for one user. Different
weights too: EXL3 at 5.05 bpw against INT4 experts here. Quality has not been
scored against each other yet ([docs/known-limits.md](docs/known-limits.md)).

| Other measurements | Value |
|---|---|
| KV offload, 64 GiB RAM tier | a 100k-token context evicted by 1.8M tokens of other traffic comes back in **0.68 s** (22.0 s without the tier); content checked correct |
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
| `SPEC_MODE` | `mtp` | `mtp` or `none` (no speculation; set `BOOT_CHECK=0`, the startup check expects drafting) |
| `SPEC_N` | `3` | MTP draft tokens per step |
| `LOCAL_ARGMAX` | `0` | `1`: per-card draft argmax, same drafts |
| `SPEC_SCHEDULE` | empty | draft count by batch size, e.g. `[[1,2,4],[3,8,3]]` with `SPEC_N=4`; not a tested serving default |
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

If this recipe helps you, please credit Morrowmake (the engine, the image and
the recipe this one descends from), halt95 (the checkpoint and its packing of
Intel's AutoRound experts, RadixArk's FP8 table and DominikBucko's MTP
packing) and Qwen (the model), and link this repository. Citation metadata:
[`CITATION.cff`](CITATION.cff).
