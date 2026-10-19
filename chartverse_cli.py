"""Prompt a local vision-language model (default: opendatalab/ChartVerse-2B) with a chart image.

The first prompt comes from --prompt-file (default: chartverse_prompt.txt next to this
script) - edit that file to iterate on the instructions; /retry re-reads it without
reloading the model. Charts are the PNGs in data/resistance_dataset/<split>/charts/, with
the OHLCV csv for the same bars in the sibling price_data/ folder (--ohlcv auto attaches it).

    python chartverse_cli.py                                            # default val chart
    python chartverse_cli.py --image data/resistance_dataset/val/charts/14_RIOT_2024-02-27.png
    python chartverse_cli.py --image <chart.png> --no-ohlcv --no-chat   # image-only, single answer

After the first answer, type follow-up questions about the same chart, or a command:
    /retry (or /r)    re-read the prompt file, fresh conversation on the same chart
    /image <path>     switch charts (fresh conversation)
    /quit (or /q)     exit
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "data" / "resistance_dataset" / "val" / "charts" / "13_PTON_2024-08-26.png"
DEFAULT_PROMPT = ROOT / "chartverse_prompt.txt"


def resolve_ohlcv(chart: Path, spec: str):
    """Return the csv text to attach, or None. spec is a path or 'auto' (sibling price_data file)."""
    if not spec:
        return None
    path = (chart.parent.parent / "price_data" / (chart.stem + ".csv")) if spec == "auto" else Path(spec)
    if not path.exists():
        print(f"note: OHLCV csv not found ({path}); continuing with the image alone")
        return None
    return path.read_text(encoding="utf8").strip()


def first_message(image, prompt_text, ohlcv_text):
    text = prompt_text.strip()
    if ohlcv_text:
        text += ("\n\nOHLCV (date,open,high,low,close,volume), oldest first; last row is the "
                 f"most recent completed bar:\n{ohlcv_text}")
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": text}]}]


def encode(processor, messages):
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                             return_dict=True, return_tensors="pt")
    except Exception as e:
        print(f"apply_chat_template return_dict failed ({e}); trying text path", flush=True)
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        imgs = [p["image"] for m in messages if isinstance(m.get("content"), list)
                for p in m["content"] if p.get("type") == "image"] or None
        return processor(text=[text], images=imgs, return_tensors="pt")


def generate(model, processor, messages, args, device):
    from transformers import TextStreamer
    inputs = encode(processor, messages).to(device)
    tokenizer = getattr(processor, "tokenizer", processor)
    n_in = inputs["input_ids"].shape[-1]
    print("", flush=True)
    t = time.time()
    gen = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0,
           "streamer": TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)}
    if gen["do_sample"]:
        gen["temperature"] = args.temperature
        gen["top_p"] = args.top_p
    if args.repetition_penalty != 1.0:
        gen["repetition_penalty"] = args.repetition_penalty
    with torch.inference_mode():
        out = model.generate(**inputs, **gen)
    new = out[0][n_in:]
    dt = time.time() - t
    print(f"\n[{new.shape[0]} new tokens in {dt:.1f}s ({new.shape[0] / dt:.1f} tok/s)]")
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def load_model(args, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    vram = f"  vram={torch.cuda.memory_allocated() / 1e9:.1f}GB" if device == "cuda" else ""
    print(f"loaded {args.model} on {device} in {time.time() - t:.1f}s{vram}", flush=True)
    return model, processor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="opendatalab/ChartVerse-2B")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    ap.add_argument("--ohlcv", nargs="?", const="auto", default=None,
                    help="attach OHLCV data: 'auto' uses the chart's sibling price_data csv, or give a csv path")
    ap.add_argument("--no-ohlcv", action="store_true", help="ignore --ohlcv even if set in a wrapper script")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="sampling temperature; 0 = greedy (greedy can degenerate into repetition loops)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="e.g. 1.1 to break repetition loops while keeping --temperature 0")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-chat", action="store_true", help="answer the prompt once and exit (no follow-ups)")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chart = Path(args.image)
    if not chart.is_absolute():
        chart = (ROOT / chart).resolve()
    if not chart.exists():
        raise SystemExit(f"chart not found: {chart}")

    model, processor = load_model(args, device)
    ohlcv_text = None if args.no_ohlcv else resolve_ohlcv(chart, args.ohlcv)

    def run_first():
        prompt_text = Path(args.prompt_file).read_text(encoding="utf8")
        print(f"chart: {chart.name}   prompt: {args.prompt_file}"
              f"   ohlcv: {'attached' if ohlcv_text else 'off'}", flush=True)
        msgs = first_message(Image.open(chart).convert("RGB"), prompt_text, ohlcv_text)
        msgs.append({"role": "assistant", "content": generate(model, processor, msgs, args, device)})
        return msgs

    messages = run_first()
    if args.no_chat:
        return
    print("\nFollow-up questions about the same chart are free (model stays loaded). "
          "Commands: /retry re-reads the prompt file, /image <path> switches chart, /q quits.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/q", "/quit", "/exit"):
            break
        if low in ("/r", "/retry"):
            messages = run_first()
        elif low.startswith("/image"):
            if len(line.split()) < 2:
                print("usage: /image <path-to-png>")
                continue
            new = Path(line.split()[1])
            if not new.is_absolute():
                new = (ROOT / new).resolve()
            if not new.exists():
                print(f"chart not found: {new}")
                continue
            chart, ohlcv_text = new, (None if args.no_ohlcv else resolve_ohlcv(new, args.ohlcv))
            messages = run_first()
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": line}]})
            messages.append({"role": "assistant",
                             "content": generate(model, processor, messages, args, device)})


if __name__ == "__main__":
    main()
