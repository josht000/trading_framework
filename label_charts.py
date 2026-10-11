"""
Interactive labeling server for resistance_dataset chart folders.

  python label_charts.py --dir data/resistance_dataset/fresh --port 8765

Open http://localhost:8765, pick a chart, then:
  - left-click on the price pane adds a horizontal level line (price snaps to cents)
  - right-click deletes the nearest added line
  - Save writes the levels to <dir>/labels.csv (sorted alphabetically by symbol, then asof)
  - Left/Right arrow keys move between charts

No dependencies beyond the project env (stdlib server + bundled plotly.min.js).
"""

import argparse
import csv
import glob
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pandas as pd
import plotly

from src.research.chart_levels import plot_levels
from src.research.config_io import make_detector
from src.research.data import load_daily

PLOTLY_JS = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="/plotly.min.js"></script>
<style>
body{font-family:system-ui,sans-serif;margin:8px}
#bar{display:flex;gap:14px;align-items:center;margin-bottom:6px}
button,a.nav{font-size:14px;padding:4px 10px}
#msg{color:#137333;font-weight:600}
.hint{color:#666;font-size:13px}
</style></head><body>
<div id="bar">
  <a class="nav" href="/">index</a>
  __PREV__
  __NEXT__
  <b>__TITLE__</b>
  <span id="cnt">0 lines</span>
  <button id="save">Save to labels.csv</button>
  <span id="msg"></span>
</div>
<div class="hint">left-click adds a level &nbsp;·&nbsp; right-click deletes the nearest &nbsp;·&nbsp; &larr;/&rarr; switch chart</div>
<div id="chart" style="width:98vw;height:86vh"></div>
<script>
var FIG = __FIG__;
var levels = __LEVELS__;
var SYM = "__SYM__", ASOF = "__ASOF__", NUM = "__NUM__";
var gd = document.getElementById('chart');
Plotly.newPlot(gd, FIG.data, FIG.layout, {scrollZoom:true, responsive:true, displaylogo:false}).then(function(){

  var drag = gd.querySelector('.nsewdrag');  // top subplot = price pane
  function redraw(){
    var yax = gd._fullLayout.yaxis;
    var shapes = FIG.layout.shapes || [];
    var ann = [];
    for(var i=0;i<levels.length;i++){
      shapes.push({type:'line', xref:'paper', x0:0, x1:1, yref:'y',
                   y0:levels[i], y1:levels[i],
                   line:{color:'rgba(31,119,180,0.95)', width:1.8, dash:'dash'}});
      ann.push({xref:'paper', x:1, xanchor:'left', yref:'y', y:levels[i],
                text: ' ' + levels[i].toFixed(2), showarrow:false,
                font:{size:11, color:'#1f77b4'}});
    }
    Plotly.relayout(gd, {shapes: shapes, annotations: ann});
    document.getElementById('cnt').textContent = levels.length + (levels.length===1?' line':' lines');
  }
  function yFromEvent(ev){
    var r = drag.getBoundingClientRect();
    var py = ev.clientY - r.top;
    if(py < 0 || py > r.height) return null;
    var yax = gd._fullLayout.yaxis;
    return yax.range[1] - (py / r.height) * (yax.range[1] - yax.range[0]);
  }
  var downX = 0, downY = 0;
  gd.addEventListener('mousedown', function(ev){ downX = ev.clientX; downY = ev.clientY; });
  gd.addEventListener('click', function(ev){
    if(Math.abs(ev.clientX - downX) + Math.abs(ev.clientY - downY) > 5) return; // was a drag
    var y = yFromEvent(ev);
    if(y === null) return;
    levels.push(Math.round(y * 100) / 100);
    levels.sort(function(a,b){return a-b});
    redraw();
    document.getElementById('msg').textContent = '';
  });
  gd.addEventListener('contextmenu', function(ev){
    ev.preventDefault();
    var y = yFromEvent(ev);
    if(y === null || !levels.length) return;
    var yax = gd._fullLayout.yaxis;
    var tol = (yax.range[1] - yax.range[0]) * 0.015;
    var bi = -1, bd = tol;
    for(var i=0;i<levels.length;i++){
      var d = Math.abs(levels[i] - y);
      if(d < bd){ bd = d; bi = i; }
    }
    if(bi >= 0){ levels.splice(bi, 1); redraw(); }
    document.getElementById('msg').textContent = '';
  });
  document.getElementById('save').addEventListener('click', function(){
    fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({symbol:SYM, asof:ASOF, num:NUM, levels:levels})})
      .then(function(r){return r.text()})
      .then(function(t){document.getElementById('msg').textContent = t;})
      .catch(function(e){document.getElementById('msg').textContent = 'save failed: ' + e;});
  });
  document.addEventListener('keydown', function(ev){
    if(ev.key === 'ArrowRight' && "__NEXTURL__") location.href = "__NEXTURL__";
    if(ev.key === 'ArrowLeft'  && "__PREVURL__") location.href = "__PREVURL__";
  });
  redraw();
});
</script></body></html>"""

INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>Label charts</title>
<style>body{font-family:system-ui,sans-serif;margin:16px}
td,th{padding:4px 10px;text-align:left} tr:nth-child(even){background:#f4f4f4}
a{font-size:14px} h3{margin:4px 0}</style></head><body>
<h3>__DIR__ — click a chart, left-click adds a level, right-click deletes, Save writes labels.csv</h3>
<table><tr><th>chart</th><th>symbol</th><th>asof</th><th>saved levels</th></tr>
__ROWS__
</table></body></html>"""


class LabelServer:
    def __init__(self, data_dir):
        self.dir = data_dir
        self.labels_csv = os.path.join(data_dir, "labels.csv")
        self.entries = []
        for p in sorted(glob.glob(os.path.join(data_dir, "price_data", "*.csv"))):
            num, sym, asof = os.path.basename(p)[:-4].split("_")
            self.entries.append({"num": num, "symbol": sym, "asof": asof})
        if not self.entries:
            raise SystemExit(f"no price_data csvs under {data_dir}")
        self.det = make_detector("atr_pivot", None)

    def read_labels(self):
        if not os.path.exists(self.labels_csv):
            return []
        with open(self.labels_csv, newline="") as f:
            return [r for r in csv.reader(f) if r and r[0] != "symbol"]

    def saved_levels(self, sym, asof):
        return sorted(float(r[2]) for r in self.read_labels()
                      if r[0] == sym and r[1] == asof and r[2].strip())

    def save(self, sym, asof, num, levels):
        rows = [r for r in self.read_labels() if not (r[0] == sym and r[1] == asof)]
        note = f"chart {int(num)}" if str(num).isdigit() else f"chart {num}"
        for lv in levels:
            rows.append([sym, asof, f"{float(lv):.2f}", note])
        if not levels:
            rows.append([sym, asof, "", note])
        def key(r):
            lv = r[2].strip()
            return (r[0].upper(), r[1], float(lv) if lv else 9e9)
        rows.sort(key=key)
        os.makedirs(os.path.dirname(self.labels_csv), exist_ok=True)
        with open(self.labels_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "asof", "level", "note"])
            w.writerows(rows)
        return len(levels)

    def chart_html(self, num):
        ent = next((e for e in self.entries if e["num"] == num), None)
        if not ent:
            return None
        sym, asof = ent["symbol"], ent["asof"]
        daily = load_daily([sym], asof, asof)[sym]
        fig = plot_levels(daily, asof, self.det, symbol=sym, lookback=160,
                          forward=15, blank=True)
        nums = [e["num"] for e in self.entries]
        i = nums.index(num)
        prev_u = f"/chart/{nums[i - 1]}" if i > 0 else ""
        next_u = f"/chart/{nums[i + 1]}" if i < len(nums) - 1 else ""
        prev_l = f'<a class="nav" href="{prev_u}">&larr; prev</a>' if prev_u else ""
        next_l = f'<a class="nav" href="{next_u}">next &rarr;</a>' if next_u else ""
        return (PAGE
                .replace("__FIG__", fig.to_json())
                .replace("__LEVELS__", json.dumps(self.saved_levels(sym, asof)))
                .replace("__TITLE__", f"chart {num} — {sym} as of {asof}")
                .replace("__SYM__", sym).replace("__ASOF__", asof)
                .replace("__NUM__", num)
                .replace("__PREV__", prev_l).replace("__NEXT__", next_l)
                .replace("__PREVURL__", prev_u).replace("__NEXTURL__", next_u))

    def index_html(self):
        rows = ""
        for e in self.entries:
            n = len(self.saved_levels(e["symbol"], e["asof"]))
            rows += (f'<tr><td><a href="/chart/{e["num"]}">{e["num"]}</a></td>'
                     f'<td>{e["symbol"]}</td><td>{e["asof"]}</td><td>{n}</td></tr>')
        return INDEX.replace("__DIR__", self.dir).replace("__ROWS__", rows)


def make_handler(srv: LabelServer):
    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype="text/html; charset=utf-8", code=200):
            data = body.encode("utf8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(srv.index_html())
            elif path == "/plotly.min.js":
                with open(PLOTLY_JS, "rb") as f:
                    self._send(f.read(), "application/javascript")
            elif m := re.fullmatch(r"/chart/(\d+)", path):
                html = srv.chart_html(m.group(1))
                self._send(html if html else "unknown chart", code=200 if html else 404)
            else:
                self._send("not found", "text/plain", 404)

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self._send("not found", "text/plain", 404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = srv.save(body["symbol"], body["asof"], str(body["num"]),
                         [float(x) for x in body["levels"]])
            self._send(f"saved {n} level(s) for {body['symbol']} {body['asof']}", "text/plain")

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join("data", "resistance_dataset", "fresh"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(LabelServer(args.dir)))
    print(f"Labeling {args.dir} at http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
