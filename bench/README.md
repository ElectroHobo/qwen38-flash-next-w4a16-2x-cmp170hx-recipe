# Benchmarks

Stdlib-only Python against a running server. Each takes a port and a model id;
run them on an idle server. The raw output of every run behind
[`docs/results.md`](../docs/results.md) is in `results/release-0.1.0/`.

| Script | Measures | Takes |
|---|---|---|
| `cmp_recipe_bench.py --port 8000 --model qwen3.8-flash-next --tag mine` | decode at 1 / 2 / 4 / 8 users and cold prefill, the MiaAI-Lab protocol Morrowmake's tables use; writes a markdown table and a JSON of the runs | ~5 min |
| `vllm_conc_steps.py 8000 qwen3.8-flash-next` | engine steps, ms per step and draft acceptance per position at 1 / 2 / 4 users, from the server's own counters | ~3 min |
| `kv_offload_bench.py mine --port 8000 --model qwen3.8-flash-next` | whether a 100k-token context evicted by a 1.8M-token flood comes back from the KV tier (time to first token, hidden codes); compare a run with `KV_OFFLOAD_GB=0` against one with the tier | ~8 min |
| `kv_dtype_check.py mine --port 8000 --model qwen3.8-flash-next` | pool size, 100k-context prefill and decode, five temperature-0 answers to diff between two KV dtypes | ~3 min |
| `prefill_fairness.py mine --port 8000 --model qwen3.8-flash-next` | what a decoding user keeps while a 30k prompt is read: tok/s before / during / after, longest pause | ~1 min |
| `soak.py` | sustained multi-turn load with image requests mixed in (from the MiMo recipe; see its header) | as configured |

`kv_offload_bench.py` sends its questions through the chat endpoint with
thinking off: this model answers a raw completion prompt with reasoning tokens
only, so the hidden-code check would read empty.
