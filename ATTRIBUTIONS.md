# Attributions

This repository contains no model weights and no copy of anyone's recipe. It runs diffbot's recipe from the user's own download
of diffbot's pack and adds the files listed in `README.md`.

| Component | Origin | Licence | Used how |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | MIT | The model. Not included. |
| EXL3 3bpw pack, sm_120 vLLM overlay, kernels, host-expert plans, `serve.sh`, benchmarks | [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) | MIT (recipe; its `patches/vllm-exl3/` is AGPL-3.0-only) | Downloaded and built by the user; `serve/serve-ssd.sh` runs its `serve.sh`. `plans/plan-1017-agent30-c512-p375.json` is derived from its plans (`plans/make_promoted_plan.py`). The `overlay/*.py` base files are the image its `docker/build.sh` produces. All benchmark and eval scripts used for the numbers are its `recipe/bench/` scripts, unchanged. |
| Engram prestage design | [nktlabs/dsv41-flash-engram-disk-prestage](https://huggingface.co/nktlabs/dsv41-flash-engram-disk-prestage) | MIT | Ported: the `prepare_inputs` hook and model resolver, `engram_prestage` / `engram_prestage_dummy` / hash parity check, fail-loud guards, step timer. Adapted to int4 tables and this vLLM base; the row reader and dequant path are new. |
| Engram-on-disk lineage | [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark), tonyd2wild | MIT | `DSV41_ENGRAM_DISK`, the eager on-disk path nktlabs' work started from. No code from it here. |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | Three modified files in `overlay/` (see `NOTICE`). |
| ExLlamaV3 / EXL3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT | Built into the image by diffbot's `docker/build.sh`. |
| Calibration method | [coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw) | MIT | diffbot's quantization follows it. |
| GSM8K, HumanEval | openai/grade-school-math, openai/human-eval | MIT | Downloaded by diffbot's `bench/fetch-evaldata.sh`. Not included. |
