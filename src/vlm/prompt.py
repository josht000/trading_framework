"""Prompt variants, the OHLCV text block, and tolerant parsing of model answers."""

import json
import math
import re
from typing import Dict, List, Optional

RUBRIC = """You are an experienced US equity swing trader reviewing a daily chart.

You are given (1) a daily candlestick chart with volume, and (2) the daily OHLCV table for exactly the bars shown in the chart, oldest first. The last row is the most recent completed bar. Use only this data; do not rely on outside knowledge of the stock.

Task: list the horizontal RESISTANCE levels the price could run into as it moves UP from the last close. A resistance level is a price ceiling where selling pressure previously stopped an advance. Only list levels ABOVE the last close.

How to identify them:
- prior swing highs where price reversed down
- areas where price was rejected at least twice; give ONE price for the area, the highest price in it
- if the stock traded in a tight range for roughly 5 to 10 days, both the highest and the lowest point of that range can act as levels
- psychological round numbers (whole dollars; half dollars for low-priced stocks) where selling clustered
- moving averages above price acting as dynamic ceilings (computable from the OHLCV table)
- high volume near a ceiling confirms it

Rules:
- Use exact prices. When a level is a prior high or low, read the price from the OHLCV table.
- Order the levels from nearest above the last close to farthest.
- Give at most 10 levels. Prefer fewer, well-supported levels over many weak ones.

Answer with JSON only, no markdown and no commentary:
{"resistance_levels":[{"price":<number>,"confidence":<0 to 1>,"reason":"<at most 12 words>"}]}"""

PLAIN = """You are given a daily candlestick chart and the daily OHLCV table for the same bars (oldest first, last row is the most recent bar).

List the horizontal resistance levels above the last close that price could run into on the way up, nearest first.

Answer with JSON only: {"resistance_levels":[{"price":<number>,"confidence":<0 to 1>,"reason":"<short>"}]}"""

FEWSHOT = """You are an experienced US equity swing trader reviewing a daily chart.

You are given (1) a daily candlestick chart with volume, and (2) the daily OHLCV table for exactly the bars shown in the chart, oldest first. The last row is the most recent completed bar. Use only this data; do not rely on outside knowledge of the stock.

Task: list the horizontal RESISTANCE levels above the last close, nearest first. A level is a zone (a prediction within $0.10 of the true level counts), but report the exact price of the extreme bar read from the OHLCV table.

Rules learned from labelled examples:
1. Report the exact `high` of the pivot bar, never a zone midpoint or a gridline guess.
2. The answer is a ladder of distinct swing highs rising from the last close: recent rejection highs, each earlier peak, then the major top or spike. Usually 3 to 6 levels.
3. Always include the highest peak in the window and sharp spike highs (long upper wick, big volume).
4. A shelf of 2 or more highs within about $0.10 to $0.15 is ONE level at the highest of them.
5. In a descending staircase of lower highs, every step is a level even when steps are only $0.10 to $0.25 apart.
6. Interior highs inside a thick supply zone count, not just its edges. If a gap wider than about $1 above price holds a cluster of highs, add the top of the densest cluster.
7. A pivot must be a clear local peak (above about 4 to 5 bars on each side); ignore one-bar noise.
8. Include the most recent high if price just rallied into it.
9. Round numbers are only a tiebreaker where highs cluster at or just under them; never invent one.
10. Drop weak near-price levels the chart barely respects; false positives there are the main error.
11. No moving averages or volume-profile levels. At most 8 levels.

Worked examples (last close, then the labelled levels):
- Last close 2.58, spike to 3.85 in May then a long fade: 2.66 (Oct highs), 2.85 (April shelf), 3.00 (March highs just under the round number), 3.24 (April pivot), 3.85 (spike top).
- Last close 17.68 after a collapse from 31: 18.26, 19.75, 23.45, 30.43, 31.56; each is the exact high of a successive lower peak down the decline.
- Last close 5.23, slow descending staircase: 5.34, 5.45, 5.59, 5.85; four lower highs only $0.10 to $0.25 apart, all kept.
- Last close 15.62 after a spike to 22.75 and a fade: 16.75, 18.30, 20.00, 22.75; the peak plus interior levels on the way down.

Answer with JSON only, no markdown and no commentary:
{"resistance_levels":[{"price":<number>,"confidence":<0 to 1>,"reason":"<at most 12 words>"}]}"""

PROMPTS = {"rubric": RUBRIC, "plain": PLAIN, "fewshot": FEWSHOT}


def packet_text(item: Dict, ohlcv_csv: str) -> str:
    """Neutral data block. The ticker is deliberately omitted to limit recall of the stock's later history."""
    return (f"Daily bars: {item['n_bars']} (from {item['first_date']} to {item['last_date']}).\n"
            f"Last bar date: {item['last_date']}. Last close: {item['price']:.2f}.\n\n"
            f"OHLCV (date,open,high,low,close,volume):\n{ohlcv_csv.strip()}")


def build_prompt(variant: str, item: Dict, ohlcv_csv: str) -> str:
    if variant not in PROMPTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {sorted(PROMPTS)}")
    return PROMPTS[variant] + "\n\n" + packet_text(item, ohlcv_csv)


def _balanced_end(text: str, start: int, open_ch: str, close_ch: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == open_ch) - (text[i] == close_ch)
        if depth == 0:
            return i
    return -1


def _json_candidates(text: str):
    for chunk in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I):
        yield chunk.strip()
    for m in re.finditer(r"[{\[]", text):
        close = "}" if m.group() == "{" else "]"
        end = _balanced_end(text, m.start(), m.group(), close)
        if end != -1:
            yield text[m.start():end + 1]


def _num(x):
    try:
        v = float(str(x).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def parse_levels(text: str) -> List[Dict]:
    """Return [{'price','confidence','reason'}]; an empty list when nothing parseable was found."""
    return parse_levels_strict(text) or []


def parse_levels_strict(text: str) -> Optional[List[Dict]]:
    """Like parse_levels, but None means no usable JSON answer (a valid empty answer returns [])."""
    for chunk in _json_candidates(text or ""):
        try:
            obj = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        items = obj
        if isinstance(obj, dict):
            items = ([obj] if "price" in obj else
                     next((obj[k] for k in ("resistance_levels", "levels", "resistance", "levels_above")
                           if k in obj), None))
        if not isinstance(items, list):
            continue
        out = []
        for it in items:
            price = _num(it.get("price") if isinstance(it, dict) else it)
            if price is None:
                continue
            conf = it.get("confidence") if isinstance(it, dict) else None
            out.append({"price": price,
                        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                        "reason": (it.get("reason") if isinstance(it, dict) else "") or ""})
        if out or items == []:
            seen, uniq = set(), []
            for lv in out:
                key = round(lv["price"], 4)
                if key not in seen:
                    seen.add(key)
                    uniq.append(lv)
            return uniq
    return None
