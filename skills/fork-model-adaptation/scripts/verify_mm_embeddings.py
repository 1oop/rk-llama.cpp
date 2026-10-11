#!/usr/bin/env python3
"""Verify multimodal embedding parity between a HF reference model and a GGUF model
served by llama-server (with an optional mmproj for vision/audio).

Generates a deterministic synthetic image and audio clip, computes HF fp32 reference
embeddings (mean pooling over the attention mask + L2 normalize), starts llama-server
with --embeddings, queries /v1/embeddings with the same inputs, and prints a cosine
similarity table with PASS/FAIL per case.

Example:
  python verify_mm_embeddings.py --hf-dir /path/to/hf-model \
      --server-bin ./build/bin/llama-server \
      --gguf model-bf16.gguf --mmproj model-mmproj-bf16.gguf

Requires: torch, transformers, numpy, pillow, soundfile (CPU is fine, just slow).
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np


def make_test_assets(tmp):
    """Deterministic synthetic image and audio, so no external files are needed."""
    from PIL import Image

    rng = np.random.RandomState(42)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, 640)[None, :]
    img[..., 1] = np.linspace(0, 255, 480)[:, None]
    img[100:200, 150:350] = [255, 0, 0]
    img[300:400, 400:600] = [0, 0, 255]
    img += rng.randint(0, 30, img.shape, dtype=np.uint8) // 4
    img_path = tmp / "test_image.png"
    Image.fromarray(img).save(img_path)

    sr = 16000
    t = np.arange(sr * 3) / sr
    wav = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    wav_path = tmp / "test_audio.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((wav * 32767).astype(np.int16).tobytes())
    return img_path, wav_path


def hf_embed_text(model, tokenizer, text):
    import torch
    import torch.nn.functional as F

    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    h = out.last_hidden_state[0].float()
    mask = enc["attention_mask"][0].bool()
    return F.normalize(h[mask].mean(dim=0), dim=0).numpy()


def hf_embed_mm(model, inputs):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state[0].float()
    mask = inputs["attention_mask"][0].bool()
    return F.normalize(h[mask].mean(dim=0), dim=0).numpy()


def start_server(args):
    cmd = [
        args.server_bin, "-m", args.gguf,
        "--embeddings", "--pooling", "mean",
        "-c", str(args.ctx), "-ub", str(args.ctx), "-b", str(args.ctx),
        "--port", str(args.port),
    ]
    if args.mmproj:
        cmd += ["--mmproj", args.mmproj]
    if args.image_max_tokens:
        cmd += ["--image-max-tokens", str(args.image_max_tokens)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{args.port}/health"
    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def server_embed_text(port, text):
    body = {"input": [text]}
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/embeddings",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return np.array(d["data"][0]["embedding"])


def server_embed_mm(port, parts):
    body = {"input": [{"content": parts}]}
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/embeddings",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return np.array(d["data"][0]["embedding"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="HF/ModelScope model directory")
    ap.add_argument("--server-bin", required=True, help="path to llama-server")
    ap.add_argument("--gguf", required=True, help="text model GGUF")
    ap.add_argument("--mmproj", default=None, help="mmproj GGUF (enables image/audio tests)")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--image-max-tokens", type=int, default=0, help="soft token budget override")
    ap.add_argument("--query-prefix", default="", help="task prefix for the short text, e.g. 'task: search result | query: '")
    ap.add_argument("--image-marker", default="<|image|>")
    ap.add_argument("--audio-marker", default="<|audio|>")
    ap.add_argument("--threshold", type=float, default=0.995, help="min cosine for PASS")
    ap.add_argument("--text-threshold", type=float, default=0.999, help="min cosine for text cases")
    args = ap.parse_args()

    import torch  # noqa: F401 - fail fast if missing
    from transformers import AutoModel, AutoTokenizer

    tmp = Path(tempfile.mkdtemp(prefix="mm_verify_"))
    img_path, wav_path = make_test_assets(tmp)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir)
    model = AutoModel.from_pretrained(args.hf_dir, dtype=torch.float32).eval()

    cases = []  # (name, hf_embed_fn, server_embed_fn, threshold)

    short_text = args.query_prefix + "What is the capital of France?"
    cases.append(("text-short", lambda: hf_embed_text(model, tokenizer, short_text),
                  lambda: server_embed_text(args.port, short_text), args.text_threshold))

    long_text = "The quick brown fox jumps over the lazy dog near the river bank. " * 300
    cases.append(("text-long-swa", lambda: hf_embed_text(model, tokenizer, long_text),
                  lambda: server_embed_text(args.port, long_text), args.text_threshold))

    if args.mmproj:
        from PIL import Image
        import soundfile as sf
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(args.hf_dir)
        img_b64 = base64.b64encode(img_path.read_bytes()).decode()
        aud_b64 = base64.b64encode(wav_path.read_bytes()).decode()
        img_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        aud_part = {"type": "input_audio", "input_audio": {"data": aud_b64, "format": "wav"}}

        img_kw = {}
        if args.image_max_tokens:
            img_kw["max_soft_tokens"] = args.image_max_tokens  # HF-side budget, gemma4-style processors

        def hf_image():
            inp = proc(text=args.image_marker, images=[Image.open(img_path).convert("RGB")],
                       return_tensors="pt", **img_kw)
            print(f"  HF image tokens: {inp['input_ids'].shape[1]}")
            return hf_embed_mm(model, inp)

        def hf_audio():
            wav, _ = sf.read(str(wav_path))
            inp = proc(text=args.audio_marker, audio=[wav], return_tensors="pt")
            print(f"  HF audio tokens: {inp['input_ids'].shape[1]}")
            return hf_embed_mm(model, inp)

        def hf_text_image():
            text_pre, text_post = "A test pattern image. ", " It has red and blue rectangles."
            inp = proc(text=text_pre + args.image_marker + text_post,
                       images=[Image.open(img_path).convert("RGB")], return_tensors="pt", **img_kw)
            print(f"  HF text+image tokens: {inp['input_ids'].shape[1]}")
            return hf_embed_mm(model, inp)

        cases.append(("image", hf_image,
                      lambda: server_embed_mm(args.port, [img_part]), args.threshold))
        cases.append(("audio", hf_audio,
                      lambda: server_embed_mm(args.port, [aud_part]), args.threshold))
        cases.append(("text+image", hf_text_image,
                      lambda: server_embed_mm(args.port, [
                          {"type": "text", "text": "A test pattern image. "},
                          img_part,
                          {"type": "text", "text": " It has red and blue rectangles."},
                      ]), args.threshold))

    server = start_server(args)
    try:
        n_fail = 0
        for name, hf_fn, srv_fn, thr in cases:
            ref = hf_fn()
            got = srv_fn()
            got = got / np.linalg.norm(got)
            cos = float(ref @ got)
            ok = cos >= thr
            n_fail += 0 if ok else 1
            print(f"{name:14s} cosine={cos:.5f} threshold={thr} {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        server.terminate()

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
