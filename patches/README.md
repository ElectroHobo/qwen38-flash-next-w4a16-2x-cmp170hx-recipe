# Engine patches

Two files from Morrowmake's [vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
at `18305948acba3f4c4d912cdd56f2202d2ebd94e8`, the commit the recipe's image is
built from, changed so that inco.ai's
[DFlash 2 drafter for Qwen3.8-Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
can attach to the model. `start.sh` and `docker-compose.yml` mount them read-only
over the image's copies when the speculation mode is `dflash` (the default); the
image itself is used as released. `SPEC_MODE=mtp` mounts neither.

| File here | Mounted over | Change |
|---|---|---|
| `qwen4_exp_nvidia_model.py` | `/opt/vllm-src/vllm/models/qwen4_exp/nvidia/model.py` | the model becomes a drafter target and captures the drafter's layers |
| `models_config.py` | `/opt/vllm-src/vllm/model_executor/models/config.py` | `dflash` accepted as a speculation method for this model |

[`qwen4exp-dflash2-capture.diff`](qwen4exp-dflash2-capture.diff) is the whole
change as a unified diff against Morrowmake's files: 48 lines added, 1 changed.

## What the change does

The engine already has everything DFlash 2 needs on the drafter side: the
speculator, the draft model and the CUDA graphs for it, which Morrowmake's GLM
recipe uses. What the Qwen3.8-Flash-Next model lacked is the other half:

- **A drafter target.** `Qwen4ExpModel` gains vLLM's `EagleModelMixin`, and the
  two model classes declare `SupportsEagle3`, the interface through which the
  engine tells a model which layers' hidden states a drafter reads.
- **The capture.** For each of those layers the layer loop hands out the
  hidden state entering it: the previous layer's pending hyper-connection
  combine completed (with that layer's `mlp_hyper_connection`, the same way the
  model's own deepstack and pipeline paths materialise it, out of place), then
  averaged over the model's four hyper-connection streams. inco.ai's capture
  patch for SGLang, shipped with the drafter, describes the training input as
  the completed, zero-based outputs of layers 5, 15, 25, 35 and 45, averaged
  over the four streams. vLLM adds one to DFlash layer ids, so the capture is
  of the state entering layers 6, 16, 26, 36 and 46 — the same tensors. The
  server logs them at start: `Using Eagle3 auxiliary layers from config: (6,
  16, 26, 36, 46)`.
- **The guard.** The model's config check refused any speculation method but
  `mtp`, `ngram` and `ngram_gpu`, which was right while nothing could feed a
  drafter; `dflash` is added.

The main computation is untouched: the capture reads the stream and averages a
copy, and only runs for the layers a drafter asked for. With DFlash 2 drafting
the quality suites score as they do with MTP (HumanEval 163/164 and GSM8K
197/200 against 161/164 and 199/200; [docs/results.md](../docs/results.md),
"DFlash 2"). `SPEC_MODE=mtp` does not mount these files at all.

## Licence

Both files are Apache-2.0, as in the engine; their licence headers are kept and
each carries a note saying what changed and where. If Morrowmake's engine gains
the same support, these mounts become unnecessary and the recipe will drop them.
