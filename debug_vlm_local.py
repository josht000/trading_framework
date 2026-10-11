"""One-off diagnostic: time each stage of a local VLM call (load, template, prefill, decode)
with a live token streamer so stalls are visible. Run:

    C:/data/envs/trading_framework/Scripts/python.exe debug_vlm_local.py [model] [--image path]
"""

import argparse
import time

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="opendatalab/ChartVerse-2B")
    ap.add_argument("--image", default="data/resistance_dataset/val/charts/13_PTON_2024-08-26.png")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer

    t = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                      attn_implementation="sdpa")
    model = model.to("cuda").eval()
    print(f"load: {time.time() - t:.1f}s  vram={torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)

    streamer = TextStreamer(processor.tokenizer, skip_special_tokens=True)

    for label, content in [
        ("text-only", [{"type": "text", "text": "Reply with the single word: ok"}]),
        ("image", [{"type": "image", "image": Image.open(args.image).convert("RGB")},
                   {"type": "text", "text": "What is the last candlestick close price? One number only."}]),
    ]:
        msgs = [{"role": "user", "content": content}]
        t = time.time()
        try:
            inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
        except Exception as e:
            print(f"[{label}] apply_chat_template return_dict failed: {e}; trying text path", flush=True)
            text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            imgs = [p["image"] for p in content if p.get("type") == "image"] or None
            inputs = processor(text=[text], images=imgs, return_tensors="pt")
        print(f"[{label}] template+encode: {time.time() - t:.1f}s "
              f"in_tokens={inputs['input_ids'].shape[-1]}", flush=True)
        inputs = inputs.to("cuda")

        t = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False, streamer=streamer)
        n = out[0][inputs["input_ids"].shape[-1]:].shape[0]
        dt = time.time() - t
        print(f"\n[{label}] gen: {dt:.1f}s for {n} tokens ({n / dt:.1f} tok/s) "
              f"vram={torch.cuda.memory_allocated() / 1e9:.1f}GB peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
              flush=True)


if __name__ == "__main__":
    main()
