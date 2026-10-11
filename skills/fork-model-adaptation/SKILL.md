---
name: fork-model-adaptation
description: End-to-end workflow for adapting a new model into this llama.cpp fork (rk-llama.cpp) and for syncing the fork with upstream ggml-org/llama.cpp. Use whenever the user wants to 适配/移植/support a model from ModelScope or HuggingFace (text, embedding, or multimodal with vision/audio towers), convert it to GGUF, verify the conversion numerically against the HF reference implementation, quantize it for deployment, or when the user wants to merge/rebase upstream llama.cpp updates into the fork (同步上游) and re-verify that adapted models still work. Also use when a converted model produces wrong embeddings, crashes during mmproj quantization, or behaves differently from the HF model.
---

# Fork model adaptation & upstream sync

This fork (rk-llama.cpp, branch `rknpu2`) tracks ggml-org/llama.cpp closely and adds the Rockchip NPU backend (`ggml/src/ggml-rknpu2`, cmake options `LLAMA_RKNPU2`/`GGML_RKNPU2`). Most "new model" requests turn out to be verification jobs, because upstream often already supports the architecture - always check before writing any code.

Read `AGENTS.md` first. Note that it governs contributions to ggml-org/llama.cpp; this fork is exempt from the upstream PR rules, but keep its code style rules (ASCII only, concise comments, no forced line wrapping) and commit conventions (`Assisted-by:`, never `Co-authored-by:`, no pushing to upstream on the user's behalf). Pushing to the user's own fork remote is fine when asked.

## Workflow A: adapt a model

### A0. Check for existing support first

Get the model's `config.json` without downloading weights, then check the `architectures` value against the conversion registry:

```bash
# ModelScope: file list and raw file download
curl -sL "https://www.modelscope.cn/api/v1/models/<org>/<name>/repo/files"
curl -sL "https://www.modelscope.cn/api/v1/models/<org>/<name>/repo?FilePath=config.json"

grep -n "<ArchitectureName>" conversion/__init__.py   # text and mmproj registries
grep -rn "<ArchitectureName>" conversion/*.py | grep register
```

Also check `src/models/` and `gguf-py/gguf/constants.py` for the arch, and search upstream PRs (`gh search prs "<model>" --repo ggml-org/llama.cpp`). If support exists, the job is: convert, verify numerically (A4), fix small issues if found. Do not re-implement an architecture that already exists.

### A1. Download the model

From ModelScope, download `config.json`, all `*.safetensors`, `tokenizer.*`, `tokenizer_config.json`, `processor_config.json`/`preprocessor_config.json` (multimodal), and for embedding models also `modules.json` and `1_Pooling/config.json` - the conversion reads pooling metadata from `modules.json` + `1_Pooling/config.json` (`_try_set_pooling_type` in `conversion/base.py`) and crashes with `FileNotFoundError` if they are missing.

Inspect the safetensors header before converting to see the real tensor names and shapes (they often differ from what the config suggests):

```python
import json, struct
with open('model.safetensors','rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
```

### A2. Convert to GGUF

Set up the python env once (CPU torch is enough):

```bash
python3 -m venv venv-gguf && venv-gguf/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv-gguf/bin/pip install transformers sentencepiece pyyaml numpy pillow soundfile torchvision --index-url ...  # torchvision needed by some image processors (e.g. gemma4)
```

```bash
# text model
venv-gguf/bin/python convert_hf_to_gguf.py <model_dir> --outfile <name>-bf16.gguf --outtype bf16
# multimodal towers (vision/audio) into an mmproj file
venv-gguf/bin/python convert_hf_to_gguf.py <model_dir> --mmproj --outfile <name>-mmproj-bf16.gguf --outtype bf16
```

Benign warnings you can ignore (upstream behavior): `Unknown RoPE type: default`, `Duplicated key name ...` (base class writes, subclass overwrites), `Unknown second separator token`. Everything else - investigate.

Check the GGUF metadata after conversion (`gguf.GGUFReader`), especially pooling type, `embedding_length_out`, `causal`, head dims, SWA pattern. Multimodal configs nest `text_config`/`vision_config`/`audio_config` - `conversion/base.py` merges `text_config` into the root hparams automatically.

### A3. Build and smoke test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-embedding llama-server llama-quantize
./build/bin/llama-embedding -m <name>-bf16.gguf -p "hello" --pooling mean   # or llama-cli for generative models
```

### A4. Verify numerically against the HF reference

This is the step that makes an adaptation trustworthy. Never declare a model "supported" without it.

- Text/generative models: use the official `examples/model-conversion/` flow (logits comparison, NMSE check, perplexity).
- Text embedding models: compare cosine similarity against the HF model run in fp32 (mean pooling over the attention mask, then L2 normalize - replicate what sentence-transformers does, including any task prompt prefixes from the model card).
- Multimodal embedding models (mmproj): use `scripts/verify_mm_embeddings.py` bundled with this skill. It generates a deterministic synthetic image and audio clip, computes HF fp32 reference embeddings, runs llama-server with the GGUF + mmproj, and prints a cosine-similarity table.

```bash
venv-gguf/bin/python <skill_dir>/scripts/verify_mm_embeddings.py \
    --hf-dir <model_dir> --server-bin ./build/bin/llama-server \
    --gguf <name>-bf16.gguf --mmproj <name>-mmproj-bf16.gguf
```

Test matrix to cover: short text, long text that crosses the SWA window (if the model has one), image only, audio only, text+image interleaved, and the quantized variants (A5). Expected agreement for a correct port, bf16 vs HF fp32: text ~0.99999, image ~0.999, audio ~0.996+. Lower than that means a real discrepancy, not rounding.

Server flags that matter for embedding models: embedding models have no KV cache, so the whole prompt (including image soft tokens) must fit in a single ubatch. Always pass `-c 8192 -ub 8192` (or larger). Otherwise the server silently caps `image_max_tokens` (watch for the `cap image_max_tokens` warning), the image gets resized to fewer tokens than HF used, and embeddings diverge.

When a modality disagrees, compare the *token counts* first (server log `n_tokens` vs HF `input_ids` length) - a count mismatch means the preprocessing (resize/expansion) differs, which is the usual cause. Compare raw cosine values only after counts match.

### A5. Quantize and re-verify

```bash
./build/bin/llama-quantize <name>-bf16.gguf <name>-q8_0.gguf Q8_0
./build/bin/llama-quantize <name>-mmproj-bf16.gguf <name>-mmproj-q8_0.gguf Q8_0
```

Re-run the A4 verification on the quantized files. Q8_0 should land within ~0.002 of the bf16 cosine. Known quantize pitfalls in this fork (fixed, but re-watch them after upstream syncs):

- 4D tensors (e.g. audio conv weights [3,3,1,C]) crashed `llama-quantize` because `nrows_total` in `src/llama-quant.cpp` ignored `ne[3]`.
- Conv weights used by `ggml_ssm_conv`/`ggml_conv_2d` (`conv_dw.weight`, `a.conv1d.*`) must stay f32; the f16 fallback crashed at runtime. They are excluded in `tensor_allows_quantization`.

### A6. Commit

Follow the repo's commit conventions: concise subject with area prefix (`mtmd : ...`, `llama-quant : ...`), short body explaining the why, `Assisted-by: <assistant name>` when the user asks you to commit. Push only to the user's fork remote. If the fix is generic (e.g. a quantize crash that also exists upstream), mention to the user that upstream likely has the same bug - contributing it upstream is their decision and must follow `AGENTS.md` (issue first, human-written PR text).

## Workflow B: sync upstream into the fork

The fork's history is squashed (single `[REBASE] Prepare for upstream NNN` commits), so rebase is impractical - use merge:

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # once
git fetch upstream
git merge upstream/master
```

Conflict resolution priorities:

1. Fork-owned files, always keep the fork version: `ggml/src/ggml-rknpu2/**`, `.devops/rknpu2.Dockerfile`, and the `LLAMA_RKNPU2`/`GGML_RKNPU2` cmake options.
2. Fork fixes on top of upstream code (e.g. the `src/llama-quant.cpp` and gemma4v preprocessor fixes): keep the fork's intent, re-apply onto the new upstream code if upstream refactored the area. Check whether upstream fixed the same bug independently - if so, drop the fork's version to reduce future conflicts.
3. Everything else: take upstream.

After merging:

1. Build both configurations: default CPU build and `-DGGML_RKNPU2=ON`.
2. Re-run the A4 verification on at least one adapted text model and one multimodal model to catch silent behavioral changes (preprocessing, attention, template changes upstream).
3. Commit the merge, push to the user's fork remote.

## Practical notes

- ModelScope raw file URL: `https://www.modelscope.cn/models/<org>/<name>/resolve/master/<path>` (works for nested paths like `1_Pooling/config.json`).
- pip may be missing on the device: `sudo apt-get install python3-pip python3-venv`.
- `pkill -f <pattern>` kills the current shell too when the pattern appears in your own command line - use `pkill -x llama-server` or kill by PID.
- When running the server repeatedly in tests, kill it between runs and wait for `/health` before sending requests.
- CPU inference of vision towers is slow in bf16 (minutes per image at 280 soft tokens on 8-core ARM); use `--image-max-tokens 70` during iteration and Q8_0 mmproj for speed, then do the final verification at the default budget.
