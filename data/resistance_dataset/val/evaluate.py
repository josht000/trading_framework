import csv, collections, sys

TOL = 0.10
pred_path = sys.argv[1] if len(sys.argv) > 1 else 'data/resistance_dataset/test/predictions.csv'

def load(path):
    d = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == 'symbol': continue
            d[(r[0], r[1])].append(float(r[2]))
    return d

labels = load('data/resistance_dataset/test/labels.csv')
preds  = load(pred_path)

tot_tp = tot_fp = tot_fn = 0
print(f"preds: {pred_path}   tolerance = +/-${TOL:.2f}\n")
print(f"{'chart':<22}{'labels':<35}{'preds':<38}{'TP':>3}{'FP':>3}{'FN':>3}")
for key in sorted(labels):
    lab = sorted(labels[key])
    prd = sorted(preds.get(key, []))
    used = set()
    tp = 0
    matches = []
    for p in prd:
        best, bi = None, -1
        for i, l in enumerate(lab):
            if i in used: continue
            if abs(p - l) <= TOL and (best is None or abs(p - l) < abs(best - p)):
                best, bi = l, i
        if bi >= 0:
            used.add(bi); tp += 1; matches.append((p, best))
    fp = len(prd) - tp
    fn = len(lab) - tp
    tot_tp += tp; tot_fp += fp; tot_fn += fn
    print(f"{key[0]+' '+key[1]:<22}{str(lab):<35}{str(prd):<38}{tp:>3}{fp:>3}{fn:>3}")
    for p, l in matches:
        print(f"    hit: pred {p:.2f} ~ label {l:.2f}  (|d|={abs(p-l):.2f})")
    for i, l in enumerate(lab):
        if i not in used:
            print(f"    miss (FN): label {l:.2f}")
    for p in prd:
        if all(p != m[0] for m in matches):
            print(f"    false pos: pred {p:.2f}")

prec = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0
rec  = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0
f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
acc  = tot_tp / (tot_tp + tot_fp + tot_fn)
print(f"\nTOTAL  TP={tot_tp}  FP={tot_fp}  FN={tot_fn}")
print(f"precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  accuracy(TP/(TP+FP+FN))={acc:.3f}")
