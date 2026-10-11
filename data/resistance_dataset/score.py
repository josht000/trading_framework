"""Score resistance predictions: python score.py <split> <predictions.csv>
Tolerance: +-$0.10, or +-1% of the label when the last close >= $50 (large caps)."""
import csv, collections, sys, os

split, pred_path = sys.argv[1], sys.argv[2]
base = os.path.dirname(os.path.abspath(__file__))

def load(path):
    d = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.reader(f):
            if r and r[0] != 'symbol':
                d[(r[0], r[1])].append(float(r[2]))
    return d

def last_close(sym, asof):
    pdir = os.path.join(base, split, 'price_data')
    fn = next(f for f in os.listdir(pdir) if f.endswith(f'_{sym}_{asof}.csv'))
    with open(os.path.join(pdir, fn)) as f:
        return float(list(csv.reader(f))[-1][4])

def score(labels, preds, flat):
    T = [0, 0, 0]
    rows = []
    for key in sorted(labels, key=lambda k: k[1]):
        lab, prd = sorted(labels[key]), sorted(preds.get(key, []))
        big = not flat and last_close(*key) >= 50
        pairs = sorted((abs(p - l), i, j) for i, p in enumerate(prd) for j, l in enumerate(lab))
        up, ul, hits = set(), set(), []
        for d, i, j in pairs:
            tol = 0.01 * lab[j] if big else 0.10
            if d <= tol + 1e-9 and i not in up and j not in ul:
                up.add(i); ul.add(j); hits.append((prd[i], lab[j]))
        tp = len(hits); fp = len(prd) - tp; fn = len(lab) - tp
        T[0] += tp; T[1] += fp; T[2] += fn
        rows.append((key, tp, fp, fn, [l for j, l in enumerate(lab) if j not in ul],
                     [p for i, p in enumerate(prd) if i not in up], hits))
    return T, rows

def report(T, name):
    tp, fp, fn = T
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    print(f"{name}: TP={tp} FP={fp} FN={fn} precision={p:.3f} recall={r:.3f} F1={f1:.3f} "
          f"accuracy(TP/(TP+FP+FN))={tp / (tp + fp + fn):.3f}")

labels, preds = load(os.path.join(base, split, 'labels.csv')), load(pred_path)
T, rows = score(labels, preds, flat=False)
print(f"{'chart':<20}{'TP':>3}{'FP':>3}{'FN':>3}  details")
for key, tp, fp, fn, miss, extra, hits in rows:
    print(f"{key[0] + ' ' + key[1]:<20}{tp:>3}{fp:>3}{fn:>3}  hits={[(round(a,2), b) for a, b in hits]} "
          f"FN={miss} FP={extra}")
report(T, "TOTAL (1% on large caps)")
report(score(labels, preds, flat=True)[0], "TOTAL (flat $0.10)")
