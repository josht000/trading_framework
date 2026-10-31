"""Input checks and per-run manifests for the strategy runner."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.daily_panel import SUFFIX


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _record(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _bars_input(root, data_dir, suffix, symbols=None):
    folder = _resolve(root, data_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"data directory not found: {folder}")
    files = {path.name[:-len(suffix)]: path for path in folder.glob(f"*{suffix}")}
    if symbols:
        requested = sorted(set(symbols))
        missing = [symbol for symbol in requested if symbol not in files]
        if missing:
            raise FileNotFoundError(f"{folder}: missing {suffix} bars for {', '.join(missing[:20])}")
        selected = [files[symbol] for symbol in requested]
    else:
        selected = [files[name] for name in sorted(files)]
    if not selected:
        raise FileNotFoundError(f"no *{suffix} daily-bar files found in {folder}")
    return {
        "directory": str(folder.resolve()),
        "suffix": suffix,
        "symbols": len(selected),
        "symbol_names": [path.name[:-len(suffix)] for path in selected],
        "files": [_record(path) for path in selected],
    }


def _rank_input(root, ranks_dir):
    folder = _resolve(root, ranks_dir)
    index_path, ranks_path = folder / "index.json", folder / "ranks.npy"
    if not index_path.is_file() or not ranks_path.is_file():
        raise FileNotFoundError(f"liquidity-rank cache needs both {index_path} and {ranks_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ranks = np.load(ranks_path, mmap_mode="r")
    expected = (len(index.get("dates", [])), len(index.get("symbols", [])))
    if ranks.shape != expected:
        raise ValueError(f"liquidity-rank matrix shape {ranks.shape} does not match index {expected}")
    return {
        "directory": str(folder.resolve()),
        "dates": expected[0],
        "symbols": expected[1],
        "symbol_names": index["symbols"],
        "files": [_record(index_path), _record(ranks_path)],
    }


def check_study_inputs(strategy, cfg, *, root=None, results_dir=None):
    """Validate configured bar/rank inputs and write a reproducibility manifest."""
    root = Path(root or Path.cwd()).resolve()
    suffix = cfg.get("data_suffix", SUFFIX)
    include_stocks = cfg.get("include_stocks", True)
    inputs = {}

    data_dir = cfg.get("data_dir")
    if data_dir and include_stocks is not False:
        symbols = [s.strip() for s in str(cfg.get("symbols", "")).split(",") if s.strip()] or None
        inputs["data_dir"] = _bars_input(root, data_dir, suffix, symbols)

    etf_data_dir = cfg.get("etf_data_dir")
    if etf_data_dir:
        symbols = [s.strip() for s in str(cfg.get("etf_symbols", "")).split(",") if s.strip()] or None
        inputs["etf_data_dir"] = _bars_input(root, etf_data_dir, suffix, symbols)

    ranks_dir = cfg.get("ranks_dir")
    if ranks_dir and include_stocks is not False:
        rank_manifest = _rank_input(root, ranks_dir)
        rank_symbols = set(rank_manifest.pop("symbol_names"))
        stock_names = set(inputs.get("data_dir", {}).get("symbol_names", []))
        missing = sorted(stock_names - rank_symbols)
        if missing:
            raise ValueError(f"liquidity-rank cache is missing stock symbols: {', '.join(missing[:20])}")
        inputs["ranks_dir"] = rank_manifest

    if strategy == "s075_residual_reversal" or cfg.get("industry_neutral"):
        simfin_dir = _resolve(root, cfg.get("simfin_dir", "data/simfin"))
        paths = [simfin_dir / "us-companies.csv", simfin_dir / "industries.csv"]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("SimFin sector-map inputs missing: " + ", ".join(map(str, missing)))
        inputs["simfin_sector_map"] = {
            "directory": str(simfin_dir.resolve()),
            "files": [_record(path) for path in paths],
        }

    if strategy == "atr_pivot":
        pit_dir = root / "data" / "pit"
        daily_paths = [pit_dir / "candidates.parquet", pit_dir / "daily_pit.parquet"]
        missing = [path for path in daily_paths if not path.is_file()]
        minute_dir = pit_dir / "alpaca_minute"
        minute_files = sorted(minute_dir.glob("*.parquet")) if minute_dir.is_dir() else []
        if missing or not minute_files:
            absent = [str(path) for path in missing]
            if not minute_files:
                absent.append(f"{minute_dir}/*.parquet")
            raise FileNotFoundError("PIT inputs missing: " + ", ".join(absent))
        inputs["pit"] = {
            "directory": str(pit_dir.resolve()),
            "minute_files": len(minute_files),
            "files": [_record(path) for path in daily_paths + minute_files],
        }

    manifest = {
        "schema_version": 1,
        "strategy": strategy,
        "variant": cfg.get("variant", "baseline"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "inputs": inputs,
        "warnings": [] if inputs else ["No supported file inputs were declared or detected for this strategy."],
    }
    if results_dir is not None:
        out = Path(results_dir) / strategy
        out.mkdir(parents=True, exist_ok=True)
        tag = cfg.get("results_tag", "")
        path = out / (f"{tag}__input_manifest.json" if tag else "input_manifest.json")
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        manifest["manifest_path"] = str(path.resolve())
    return manifest
