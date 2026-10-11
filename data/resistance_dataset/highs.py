"""Price lookup helper: python highs.py <csv> [window]
Prints last close and every bar high above it that is the max of +-window bars (default 2),
so the reader can pick exact pivot prices. It does not choose levels."""
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
c = float(rows[-1]['close'])
print(f"last close {c}  bars {len(rows)}  {rows[0]['date']}..{rows[-1]['date']}")
H = [float(r['high']) for r in rows]
for i, r in enumerate(rows):
    if H[i] > c and H[i] == max(H[max(0, i - w):i + w + 1]):
        print(f"{i:3d} {r['date']} H={H[i]:.4f} O={float(r['open']):.4f} "
              f"L={float(r['low']):.4f} C={float(r['close']):.4f} "
              f"V={int(float(r['volume'])) // 1000}k")
