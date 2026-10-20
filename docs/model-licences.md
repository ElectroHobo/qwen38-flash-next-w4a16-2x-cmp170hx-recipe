# Model and checkpoint licences

This recipe downloads weights it does not own. Read the terms before serving
them to anyone else.

| Component | Where it comes from | Licence |
|---|---|---|
| Qwen3.8-Flash-Next (the model) | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Qwen Community License 1.0 (the checkpoint ships it as `LICENSE`) |
| The checkpoint served here | [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin), revision `cec53e9f` | the card says `license: other`, Qwen Community License 1.0 |
| Routed experts, INT4 AutoRound | [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound), repacked by halt95 | as the Intel card states |
| FP8 n-gram (PLE) table | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), repacked by halt95 | as the RadixArk card states |
| MTP draft head, INT4-packed | DominikBucko's packing recipe, applied by halt95 | inside the checkpoint, same terms |
| The DFlash 2 drafter (the default) | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision `a22a1744` | Apache-2.0 |
| The two patched engine files in `patches/` | Morrowmake's vllm-cmp170hx, modified | Apache-2.0, headers kept, changes noted |
| The engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) | Apache-2.0 |
| The engine image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6…` | Apache-2.0, with the image's `THIRD_PARTY_NOTICES` |
| This recipe's scripts | Morrowmake's Qwen3.8-Flash-Next recipe, modified | MIT |
| Benchmark prompts | MiaAI-Lab's kits, used as data with attribution | see [results.md](results.md), "Notes on the method" |

inco.ai's Qwen drafter is Apache-2.0 — unlike their GLM drafter, which is CC
BY-NC-ND, so check the licence before swapping drafters.

halt95's model card credits Qwen, Intel, RadixArk, DominikBucko, the vLLM
community's PLE-offload work and aikitoria; this recipe passes those credits
on. If you redistribute weights derived from this checkpoint, the Qwen
Community License's attribution and naming terms apply to you.
