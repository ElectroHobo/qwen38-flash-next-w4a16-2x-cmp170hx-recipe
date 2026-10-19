# Status and known limits

## Status

Release 0.1.0 (2026-10-19). Measured on one box, one weekend: a dual Xeon
E5-2697A v4 host with two CMP 170HX cards. The serving configuration ran
through the decode, prefill, step, KV offload and long-context benchmarks in
[results.md](results.md) and served an Open WebUI and agent sessions for the
rest of the day without an engine error. Nothing here has run for a week yet.

## Known limits

- **No quality score yet.** The checkpoint's own card gates it against the BF16
  model on halt95's engine (GSM8K 198/200, structured output 80/80, a 262K
  needle). On this engine, with the FP8 KV scales unused and the table in RAM,
  the same weights have only been spot-checked: identical code at temperature
  0 to the same checkpoint on another engine, correct answers to the offload
  bench's hidden-code questions, and a day of use. A HumanEval / GSM8K pass
  rate against Morrowmake's EXL3 recipe is the open item before this is trusted
  as more than fast. Until then, treat the EXL3 recipe as the quality reference
  and this one as the speed reference.
- **INT4 experts, not EXL3.** The EXL3 recipe's weights are 5.05 bits per
  weight; this checkpoint's experts are 4-bit with 128-wide groups. If that
  difference matters to you, it is the one above.
- **96 GB of host RAM is the floor.** The n-gram table is 51 GB of pinned host
  memory, allocated at start; the engine refuses to start without it. The EXL3
  recipe streams the same table from the SSD and runs in 32 GB.
- **Two cards, whole experts per card.** Tensor-parallel 2 only. An expert
  cannot be split across the cards (the INT4 scale groups would straddle them),
  and no other layout is wired up.
- **The first start compiles for about 15 minutes.** Later starts reuse
  `cache/`; a change of `GRAPH_MODE` or the KV dtype recompiles part of it.
- **Decode is power-bound at 180 W.** With three drafts the cards sit at the
  cap during decode at 1400 to 1499 MHz. Locking clocks does nothing; raising
  the cap is a decision about your power supply, not this recipe.
- **The disk tier fills disks.** `KV_OFFLOAD_DIR` is write-through and never
  pruned. One benchmark run wrote 58 GB. A dedicated volume with a hard size, or
  leave it off.
- **FP8 KV costs decode on these cards** (no FP8 units): −8 to −10 % at one
  user for a 73 % larger pool, and the answers shift. Documented, not default.
- **halt95's drafter is MTP; the DFlash 2 drafter is not wired in.**
  [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  (Apache-2.0, published 2026-10-19) reports +23 % over MTP at one user on its
  authors' engine. Morrowmake's engine has the DFlash 2 speculator but no
  capture hook for this model yet; it needs a port of inco.ai's small SGLang
  patch to the engine's Qwen model. Not started.
- **Agent fairness is not tuned.** A 30k-token prompt arriving while someone is
  decoding stalls their decode for the read (about 7 s at 4,500 tok/s). The
  engine's `--prefill-chunk-with-decodes` is the lever and has not been
  measured on this build; `MAX_BATCHED` is the blunt one.
- **The engine image is Morrowmake's and may move.** This recipe pins it by
  digest and runs it unchanged; if they retire the digest, `./start.sh update`
  cannot help until this recipe is re-pinned and re-measured.
- **`docker/` is Morrowmake's build tooling, unchanged** beyond the
  checkpoint's name and pin in `prepare.sh` and `container.env`. It still names
  the EXL3 checkpoint in the image build files; this recipe does not rebuild
  the image.
