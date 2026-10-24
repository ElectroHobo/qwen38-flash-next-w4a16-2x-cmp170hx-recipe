# Status and known limits

## Status

Release 0.2.0 (2026-10-23). Measured on one box: a dual Xeon E5-2697A v4 host
with two CMP 170HX cards. 0.1.0 was measured over a weekend; 0.2.0's
additions (RecoverSSM, BF16 GDN states, 128 requests, the shared table, the MTP
schedule) ran through a day of development benchmarks, debug checks and quality
runs, then the release benchmarks from a fresh clone. They have not served
real traffic for days yet.

## Known limits

- **Quality is scored on two checkable suites only.** HumanEval 162/164 and
  GSM8K 198/200 (first 200 problems) with the DFlash 2 default, 162/164 and
  197/200 with MTP, thinking on ([results.md](results.md), "Release 0.2.0"). The checkpoint's own card adds GSM8K
  198/200, structured output 80/80 and a 262K needle on halt95's engine.
  Nothing here measures long-form writing, tool-call adherence or hard
  competition code.
- **INT4 experts, not EXL3.** The EXL3 recipe's weights are 5.05 bits per
  weight; this checkpoint's experts are 4-bit with 128-wide groups. The two
  suites above are not where such a difference would show first.
- **96 GB of host RAM is the floor.** The n-gram table is 48 GiB of pinned host
  memory (64 GiB as the engine allocates it), allocated at start; the engine
  refuses to start without it. The EXL3 recipe streams the same table from the
  SSD and runs in 32 GB. `PLE_SHARED_DIR` shares one copy between servers; its
  file stays in RAM until a reboot or until you delete it.
- **Two cards, whole experts per card, tensor parallel only.** An expert
  cannot be split across the cards (the INT4 scale groups would straddle them),
  and the engine refuses pipeline parallelism for this model ("N-gram PLE
  embedding requires pipeline_parallel_size=1"). That is why this recipe needs
  x16 links and peer-to-peer where AlexBrandes's pipeline-parallel build runs
  on stock x4 cards, and why its prompt reading is slower: tensor parallel
  moves an activation between the cards at every layer. Pipeline parallel on
  this engine would be a port of their patches.
- **The first start compiles for about 22 minutes.** That includes RecoverSSM's
  CUDA verify (about 95 s, built with the image's compiler into `cache/`). Later
  starts reuse `cache/`; a change of `GRAPH_MODE`, the drafter, `MAX_SEQS` or the
  KV dtype recompiles part of it. From a slow disk (a USB drive, a network share)
  loading the 116 GB checkpoint adds to that and the first start can pass the
  30-minute `READY_TIMEOUT`; set `READY_TIMEOUT=7200` in `.env` for it.
- **Decode is power-bound at 180 W.** With three MTP drafts the cards already
  sat at the cap during decode, at 1400 to 1499 MHz. Locking clocks does nothing; raising
  the cap is a decision about your power supply, not this recipe.
- **The disk tier fills disks.** `KV_OFFLOAD_DIR` is write-through and never
  pruned. One benchmark run wrote 58 GB. A dedicated volume with a hard size, or
  leave it off.
- **FP8 KV costs decode on these cards** (no FP8 units): −8 to −10 % at one
  user for a 73 % larger pool, and the answers shift. Documented, not default.
- **The engine runs with patched files.** Morrowmake's image is used as
  released, but `start.sh` and Compose mount source files from
  [`patches/`](../patches/) over the image's: the DFlash 2 capture (dflash
  mode), RecoverSSM, the n-gram table loader and the chunked copy-on-write (every
  mode). They are pinned to the engine commit this recipe pins; a new engine
  release needs them re-made against it.
- **RecoverSSM is ours, not the engine's.** It follows the engine's own design
  for GLM's KDA layers and was checked bitwise against the path it replaces
  ([`patches/README.md`](../patches/README.md)), but it is new code on a recurrent
  state the model depends on. `RECOVER_SSM=0` is the way back.
- **Seven drafts need a chosen block size.** Five to seven DFlash drafts need an
  attention block size the sparse-attention ring (12) and the prefix-match unit
  (16) divide, at least the GDN state page: 912 with BF16 states, 1680 with
  float32, set by `serve.sh`. A size too small for the state page is silently
  doubled by the engine, halving how many requests fit. The drafter allows at
  most 7.
- **Many users get the total, not the speed.** At 128 requests the total is
  ~1,950 tok/s but each user gets ~15; MTP is ahead of DFlash 2 from about 32
  users and behind it below, and one server cannot switch between them.
- **Follow-up turns are a little slower than with MTP.** The second turn of an
  18.7k-token conversation starts in 0.42 s, against 0.25 s with MTP: four or
  more drafts need a prefix-match unit of 16 instead of 64.
- **Agent fairness is not tuned.** A 30k-token prompt arriving while someone is
  decoding drops their decode to about 8 to 10 tok/s for the 6 s read. The
  engine's `--prefill-chunk-with-decodes 384` (via `EXTRA_ARGS`) keeps 13 tok/s
  and cuts the longest pause from 0.7 to 0.25 s, but stretches the prompt's read
  from 6.3 to 22.7 s, so it is not the default.
- **The engine image is Morrowmake's and may move.** This recipe pins it by
  digest and runs it as released, with the files above mounted; if they
  retire the digest, `./start.sh update`
  cannot help until this recipe is re-pinned and re-measured.
- **`docker/` is Morrowmake's build tooling, unchanged** beyond the
  checkpoint's name and pin in `prepare.sh` and `container.env`. It still names
  the EXL3 checkpoint in the image build files; this recipe does not rebuild
  the image.
