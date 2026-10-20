# Status and known limits

## Status

Release 0.1.0 (2026-10-19). Measured on one box, one weekend: a dual Xeon
E5-2697A v4 host with two CMP 170HX cards. The serving configuration ran
through the decode, prefill, step, KV offload and long-context benchmarks in
[results.md](results.md) and served an Open WebUI and agent sessions for the
rest of the day without an engine error; the DFlash 2 default served from the
evening on. Nothing here has run for a week yet.

## Known limits

- **Quality is scored on two checkable suites only.** HumanEval 163/164 and
  GSM8K 197/200 (first 200 problems) with the DFlash 2 default, 161/164 and
  199/200 with MTP, thinking on, no reply cut off ([results.md](results.md),
  "Quality"). The checkpoint's own card adds GSM8K
  198/200, structured output 80/80 and a 262K needle on halt95's engine.
  Nothing here measures long-form writing, tool-call adherence or hard
  competition code.
- **INT4 experts, not EXL3.** The EXL3 recipe's weights are 5.05 bits per
  weight; this checkpoint's experts are 4-bit with 128-wide groups. The two
  suites above are not where such a difference would show first.
- **96 GB of host RAM is the floor.** The n-gram table is 51 GB of pinned host
  memory, allocated at start; the engine refuses to start without it. The EXL3
  recipe streams the same table from the SSD and runs in 32 GB.
- **Two cards, whole experts per card, tensor parallel only.** An expert
  cannot be split across the cards (the INT4 scale groups would straddle them),
  and the engine refuses pipeline parallelism for this model ("N-gram PLE
  embedding requires pipeline_parallel_size=1"). That is why this recipe needs
  x16 links and peer-to-peer where AlexBrandes's pipeline-parallel build runs
  on stock x4 cards, and why its prompt reading is slower: tensor parallel
  moves an activation between the cards at every layer. Pipeline parallel on
  this engine would be a port of their patches.
- **The first start compiles for about 15 minutes.** Later starts reuse
  `cache/`; a change of `GRAPH_MODE` or the KV dtype recompiles part of it.
- **Decode is power-bound at 180 W.** With three MTP drafts the cards already
  sat at the cap during decode, at 1400 to 1499 MHz. Locking clocks does nothing; raising
  the cap is a decision about your power supply, not this recipe.
- **The disk tier fills disks.** `KV_OFFLOAD_DIR` is write-through and never
  pruned. One benchmark run wrote 58 GB. A dedicated volume with a hard size, or
  leave it off.
- **FP8 KV costs decode on these cards** (no FP8 units): −8 to −10 % at one
  user for a 73 % larger pool, and the answers shift. Documented, not default.
- **DFlash 2 runs on two patched engine files.** Morrowmake's image is used as
  released, but in the default mode `start.sh` mounts two of its source files
  from [`patches/`](../patches/) over the image's (48 lines added, 1 changed).
  They are pinned to the engine commit this recipe pins; a new engine release
  needs them re-made against it. `SPEC_MODE=mtp` runs the image unmodified.
- **Seven drafts need a forced block size.** Five to seven DFlash drafts start
  only with `--block-size 1680`, which `serve.sh` sets; a different `SPEC_N` or
  context is untested with it. The drafter allows at most 7.
- **Follow-up turns are a little slower than with MTP.** The second turn of an
  18.7k-token conversation starts in 0.49 s, against 0.25 s with MTP: four or
  more drafts need a prefix-match unit of 16 instead of 64.
- **Agent fairness is not tuned.** A 30k-token prompt arriving while someone is
  decoding drops their decode to about 8 to 10 tok/s for the 6 s read. The
  engine's `--prefill-chunk-with-decodes 384` (via `EXTRA_ARGS`) keeps 13 tok/s
  and cuts the longest pause from 0.7 to 0.25 s, but stretches the prompt's read
  from 6.3 to 22.7 s, so it is not the default.
- **The engine image is Morrowmake's and may move.** This recipe pins it by
  digest and runs it as released, with the two files above mounted; if they
  retire the digest, `./start.sh update`
  cannot help until this recipe is re-pinned and re-measured.
- **`docker/` is Morrowmake's build tooling, unchanged** beyond the
  checkpoint's name and pin in `prepare.sh` and `container.env`. It still names
  the EXL3 checkpoint in the image build files; this recipe does not rebuild
  the image.
