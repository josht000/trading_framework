"""
Minimal OpenAI-compatible /v1/chat/completions server for local HF vision-language models.

  python -m src.vlm.serve_local --model opendatalab/ChartVerse-2B --port 8081
  python -m src.vlm.serve_local --model AITRADER/Amsi-fin-o1 --port 8082 --trust-remote-code
  python -m src.vlm.serve_local --model tjarvis91/vfai-x-3.5-9b-options --port 8083 --device cpu

Implements only what src.vlm.providers needs: POST /v1/chat/completions with messages whose
content parts are {type: text} and {type: image_url} where image_url is a data:...;base64 URL.
Generation is serialized (one request at a time) and temperature 0 means greedy decoding.
"""

import argparse
import base64
import io
import threading
import time

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()
STATE = {"model": None, "processor": None, "device": "cpu", "model_id": "", "lock": threading.Lock(),
         "max_new_cap": 4096, "min_pixels": None, "max_pixels": None}


def _decode_image(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only data: URLs are supported for local inference")
    _, _, payload = url.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _template_messages(messages):
    """Convert OpenAI content parts to template parts; return (messages, [PIL images])."""
    out, images = [], []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append({"role": m.get("role", "user"), "content": content or ""})
            continue
        parts = []
        for p in content:
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") in ("image_url", "image"):
                url = (p.get("image_url") or {}).get("url") or p.get("url") or ""
                img = _decode_image(url)
                images.append(img)
                parts.append({"type": "image", "image": img})
        out.append({"role": m.get("role", "user"), "content": parts})
    return out, images


def _qwen_vl_prompt(messages) -> str:
    """Manual ChatML/Qwen-VL rendering for checkpoints whose processor lacks a chat template."""
    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            body = "".join("<|vision_start|><|image_pad|><|vision_end|>" if p.get("type") == "image"
                           else p.get("text", "") for p in content)
        else:
            body = content or ""
        chunks.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["model_id"], "device": STATE["device"]}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": STATE["model_id"], "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages, images = _template_messages(body.get("messages", []))
    max_new = body.get("max_tokens") or body.get("max_completion_tokens") or STATE["max_new_cap"]
    max_new = min(int(max_new), STATE["max_new_cap"])
    temperature = body.get("temperature")
    do_sample = temperature is not None and float(temperature) > 0

    processor, model = STATE["processor"], STATE["model"]
    pix = {k: v for k, v in (("min_pixels", STATE["min_pixels"]), ("max_pixels", STATE["max_pixels"]))
           if v is not None}
    print(f"[serve] request: {len(messages)} messages, {len(images)} images, max_new={max_new}", flush=True)
    try:
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", **pix)
    except Exception:
        try:
            tok = getattr(processor, "tokenizer", processor)
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            text = _qwen_vl_prompt(messages)
        inputs = processor(text=[text], images=images or None, return_tensors="pt", **pix)
    inputs = inputs.to(STATE["device"])

    gen_kwargs = {"max_new_tokens": max_new, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        if body.get("top_p") is not None:
            gen_kwargs["top_p"] = float(body["top_p"])

    with STATE["lock"]:
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        latency = time.time() - t0

    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    n_in, n_out = int(inputs["input_ids"].shape[-1]), int(new_ids.shape[0])
    print(f"[serve] in={n_in} out={n_out} {latency:.1f}s", flush=True)
    return JSONResponse({
        "id": "local-" + str(int(t0)),
        "object": "chat.completion",
        "model": body.get("model") or STATE["model_id"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "length" if n_out >= max_new else "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
        "latency_s": round(latency, 2),
    })


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes 4-bit quantization (needs CUDA; falls back to dtype if unavailable)")
    ap.add_argument("--max-new-cap", type=int, default=2048,
                    help="cap on max_new_tokens per request (thinking models can ramble)")
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="image pixel floor for Qwen-style processors (e.g. 3136)")
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="image pixel cap for Qwen-style processors (e.g. 802816=1024*28*28) to bound vision tokens")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype(args.dtype)

    from transformers import AutoProcessor
    loaders = ["AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"]
    kw = {"trust_remote_code": args.trust_remote_code or None}
    kw = {k: v for k, v in kw.items() if v is not None}

    processor = AutoProcessor.from_pretrained(args.model, **kw)
    model, last_err = None, None
    if args.load_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                           bnb_4bit_compute_dtype=dtype)
            kw["device_map"] = "auto"
            print("loading with bitsandbytes 4-bit", flush=True)
        except ImportError:
            print("bitsandbytes not installed; loading full precision", flush=True)
    for loader in loaders:
        try:
            cls = getattr(__import__("transformers"), loader)
            model = cls.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa", **kw)
            break
        except (AttributeError, ValueError, OSError, KeyError) as e:
            last_err = e
    if model is None:
        raise SystemExit(f"could not load {args.model}: {last_err}")
    if args.load_4bit and "quantization_config" in kw:
        model = model.eval()  # quantized models are already placed by device_map
    else:
        model = model.to(device).eval()

    STATE.update(model=model, processor=processor, device=device, model_id=args.model,
                 max_new_cap=args.max_new_cap, min_pixels=args.min_pixels,
                 max_pixels=args.max_pixels)
    print(f"serving {args.model} on {device}/{args.dtype} at http://{args.host}:{args.port}", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws="none")


if __name__ == "__main__":
    main()
