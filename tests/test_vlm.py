import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.research.chart_png import render_chart_png
from src.vlm.estimate import estimate_call, model_table
from src.vlm.fewshot import load_shots, load_split_items, shot_answer, shots_fingerprint
from src.vlm.models import load_registry, select_models
from src.vlm.packets import build_packets, load_manifest
from src.vlm.prompt import build_prompt, build_system_prompt, packet_text, parse_levels, parse_levels_strict
from src.vlm.providers import (ProviderError, build_anthropic_payload, build_chat_payload, build_openai_payload,
                               call_model, mock_spec, parse_anthropic, parse_chat, parse_openai)
from src.vlm.report import build_report
from src.vlm.runner import load_runs, missing_keys, run_benchmark, run_key
from src.vlm.scoring import leaderboard, score_levels, score_runs

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100
DATASET_ROOT = os.path.join("data", "resistance_dataset")


def fake_item(idx=0):
    return {"id": f"{idx:02d}_AAA_2024-01-10", "symbol": "AAA", "asof": "2024-01-10", "price": 5.0, "atr14": 0.4,
            "n_bars": 100, "first_date": "2023-08-01", "last_date": "2024-01-10", "gold_levels": [5.5, 6.2],
            "chart_blind": "blind/01.png", "chart_labeled": "labeled/01.png", "ohlcv": "ohlcv/01.csv"}


def make_root(items):
    root = tempfile.mkdtemp()
    for sub in ("blind", "labeled", "ohlcv"):
        os.makedirs(os.path.join(root, sub))
    for it in items:
        with open(os.path.join(root, it["chart_blind"]), "wb") as f:
            f.write(PNG)
        with open(os.path.join(root, it["ohlcv"]), "w", encoding="utf8") as f:
            f.write("date,open,high,low,close,volume\n2024-01-10,4.9,5.1,4.8,5.0,1000\n")
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf8") as f:
        json.dump({"created": "x", "bars": 99, "image_px": [1600, 1000], "items": items}, f)
    return root


class TestPrompt(unittest.TestCase):
    def test_parse_variants(self):
        lv = parse_levels('{"resistance_levels":[{"price":5.5,"confidence":0.9,"reason":"swing"},{"price":6.2}]}')
        self.assertEqual([x["price"] for x in lv], [5.5, 6.2])
        self.assertEqual(parse_levels('```json\n{"resistance_levels":[{"price":3.1}]}\n```')[0]["price"], 3.1)
        self.assertEqual(parse_levels('sure! [{"price":1.0}] next [{"price":2.0}]')[0]["price"], 1.0)
        self.assertEqual(parse_levels("[5.5, 6.2]"), [{"price": 5.5, "confidence": None, "reason": ""},
                                                    {"price": 6.2, "confidence": None, "reason": ""}])
        self.assertEqual([x["price"] for x in parse_levels('{"resistance_levels":[{"price":"$5.50"},{"price":"abc"}]}')],
                         [5.5])

    def test_strict_distinguishes_empty_from_garbage(self):
        self.assertEqual(parse_levels_strict('{"resistance_levels":[]}'), [])
        self.assertIsNone(parse_levels_strict("I cannot answer."))
        self.assertEqual(parse_levels("I cannot answer."), [])

    def test_build_prompt_contains_no_symbol_or_labels(self):
        text = build_prompt("rubric", fake_item(), "date,open,high,low,close,volume\n...")
        self.assertNotIn("AAA", text)
        self.assertNotIn("gold", text.lower())
        self.assertIn("5.00", text)

    def test_system_prompt_and_packet_text(self):
        system = build_system_prompt("learned", "Prefer exact swing highs.")
        self.assertIn("worked examples", system)
        self.assertIn("Prefer exact swing highs.", system)
        self.assertIn("final user message", system)
        packet = packet_text(fake_item(), "date,open,high,low,close,volume\n...", "watch the upper shelf")
        self.assertIn("Analyst notes for this chart", packet)
        self.assertIn("watch the upper shelf", packet)


class TestProviders(unittest.TestCase):
    def test_payload_shapes(self):
        spec = mock_spec("empty")
        png = PNG
        a = build_anthropic_payload(spec, "hi", png, 100)
        self.assertEqual(a["messages"][0]["content"][0]["type"], "image")
        o = build_openai_payload(spec, "hi", png, 100)
        self.assertEqual(o["input"][0]["content"][0]["type"], "input_image")
        c = build_chat_payload(spec, "hi", png, 100)
        self.assertIn("base64", c["messages"][0]["content"][0]["image_url"]["url"])

    def test_fewshot_multiturn_order(self):
        spec = mock_spec("empty")
        shot = {"png": PNG, "text": "SHOT_TEXT", "answer": "SHOT_ANSWER"}
        c = build_chat_payload(spec, "QUERY", PNG, 100, "SYSTEM", [shot])
        self.assertEqual(c["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(c["messages"][1]["role"], "user")
        self.assertEqual(c["messages"][1]["content"][0]["type"], "image_url")
        self.assertEqual(c["messages"][1]["content"][1]["text"], "SHOT_TEXT")
        self.assertEqual(c["messages"][2], {"role": "assistant", "content": "SHOT_ANSWER"})
        self.assertEqual(c["messages"][3]["content"][1]["text"], "QUERY")

        o = build_openai_payload(spec, "QUERY", PNG, 100, "SYSTEM", [shot])
        self.assertEqual(o["input"][0]["role"], "developer")
        self.assertEqual(o["input"][1]["content"][0]["type"], "input_image")
        self.assertEqual(o["input"][2]["role"], "assistant")
        self.assertEqual(o["input"][3]["content"][0]["type"], "input_image")
        self.assertEqual(o["input"][3]["content"][1]["text"], "QUERY")

        a = build_anthropic_payload(spec, "QUERY", PNG, 100, "SYSTEM", [shot])
        self.assertEqual(a["system"], "SYSTEM")
        self.assertEqual(a["messages"][0]["content"][0]["type"], "image")
        self.assertEqual(a["messages"][1]["role"], "assistant")
        self.assertEqual(a["messages"][2]["content"][1]["text"], "QUERY")

    def test_single_query_packet_enforced(self):
        with self.assertRaises(ProviderError):
            call_model(mock_spec("empty"), "x", [PNG, PNG], "", item=fake_item())

    def test_parsers(self):
        r = parse_anthropic({"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn",
                             "usage": {"input_tokens": 3, "output_tokens": 4}})
        self.assertEqual((r.text, r.input_tokens, r.output_tokens, r.finish_reason), ("x", 3, 4, "end_turn"))
        r = parse_openai({"output": [{"type": "message", "content": [{"type": "output_text", "text": "y"}]}],
                          "status": "completed",
                          "usage": {"input_tokens": 1, "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 1}}})
        self.assertEqual((r.text, r.reasoning_tokens, r.finish_reason), ("y", 1, "completed"))
        r = parse_chat({"choices": [{"message": {"content": "z"}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.001}})
        self.assertEqual((r.text, r.reported_cost, r.finish_reason), ("z", 0.001, "length"))


class TestFewShot(unittest.TestCase):
    def test_split_items_load_from_dataset(self):
        train, val, test = (load_split_items(s, DATASET_ROOT) for s in ("train", "val", "test"))
        self.assertEqual((len(train), len(val), len(test)), (11, 8, 20))
        self.assertEqual(val[0]["id"], "13_PTON_2024-08-26")
        for it in train + val + test:
            self.assertTrue(os.path.exists(it["chart_path"]))
            self.assertTrue(os.path.exists(it["ohlcv_path"]))
            self.assertTrue(np.isfinite(it["atr14"]))
            self.assertTrue(it["gold_levels"])

    def test_shot_answer_and_load_shots(self):
        item = fake_item()
        answer = shot_answer(item, "analyst reasoning")
        self.assertTrue(answer.startswith("analyst reasoning"))
        parsed = parse_levels_strict(answer)
        self.assertEqual([x["price"] for x in parsed], [5.5, 6.2])
        shots = load_shots(("train",), DATASET_ROOT)
        self.assertEqual(len(shots), 11)
        self.assertTrue(all(s["png"].startswith(b"\x89PNG") for s in shots))
        self.assertTrue(all("resistance_levels" in s["answer"] for s in shots))
        self.assertEqual(shots_fingerprint(shots), shots_fingerprint(load_shots(("train",), DATASET_ROOT)))
        with self.assertRaises(ValueError):
            load_shots(("test",), DATASET_ROOT)


class TestRegistryAndEstimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = load_registry()

    def test_registry_valid(self):
        self.assertGreaterEqual(len(self.reg["models"]), 10)
        for m in self.reg["models"]:
            self.assertGreater(m.price_in, 0)
            self.assertTrue(m.key_env or m.route == "mock")

    def test_deepseek_registry_entries(self):
        by_key = {m.key: m for m in self.reg["models"]}
        for key in ("deepseek-v4.1-flash", "deepseek-v4-flash-vision-exp"):
            self.assertIn(key, by_key)
            self.assertEqual(by_key[key].extra_body["reasoning"]["effort"], "low")
            self.assertEqual(by_key[key].est_reasoning_tokens, 3000)
            self.assertTrue(by_key[key].first_pass)

    def test_select_and_via(self):
        first = select_models(self.reg, None, True)
        self.assertTrue(all(m.first_pass for m in first))
        via_or = first[0].via("openrouter")
        self.assertEqual(via_or.route, "openrouter")
        self.assertTrue(via_or.api_model)

    def test_estimate_cost(self):
        from src.vlm.models import ModelSpec
        spec = ModelSpec(key="t", label="t", vendor="t", tier="mid", route="openrouter", api_model="x", key_env="K",
                         openrouter_id="x", price_in=1.0, price_out=2.0, est_reasoning_tokens=0, context="")
        e = estimate_call(spec, 1000, 1000, 4000, "low")
        expected = (e["input_tokens"] * 1.0 + e["output_tokens"] * 2.0) / 1e6
        self.assertAlmostEqual(e["cost"], expected)
        e2 = estimate_call(spec, 1000, 1000, 4000, "low", system_chars=1000, n_images=3)
        self.assertGreater(e2["input_tokens"], e["input_tokens"])
        df = model_table([spec], 1000, 1000, 4000, n_charts=2, repeats=1, system_chars=1000, n_images=3)
        self.assertAlmostEqual(df.iloc[0]["est_$/run(2x1)"], e2["cost"] * 2, places=6)


class TestScoring(unittest.TestCase):
    def test_exact_and_tolerance(self):
        s = score_levels([5.5, 6.2], [5.5, 6.2], price=5.0, atr=0.4)
        self.assertEqual((s["tp@0.5"], s["f1@0.5"]), (2, 1.0))
        s = score_levels([5.5], [5.5 + 0.3 * 0.4], price=5.0, atr=0.4)   # 0.3 ATR away
        self.assertEqual((s["tp@0.25"], s["tp@0.5"]), (0, 1))
        s = score_levels([5.5], [4.9], price=5.0, atr=0.4)               # pred below price is ignored
        self.assertEqual(s["n_pred"], 0)
        self.assertEqual(s["n_pred_below_price"], 1)
        self.assertEqual(score_levels([5.5], [], 5.0, 0.4)["recall@0.5"], 0.0)

    def test_leaderboard_aggregation(self):
        runs = [{"model": "m1", "item": "i1", "repeat": 0, "status": "ok", "levels": [{"price": 5.5}], "cost_usd": 0.01,
                 "latency_s": 1.0, "input_tokens": 1, "output_tokens": 2, "split": "val", "shots": "train"},
                {"model": "m1", "item": "i2", "repeat": 0, "status": "error", "error": "boom", "cost_usd": 0.0,
                 "split": "val", "shots": "train"}]
        items = {"i1": fake_item(1), "i2": fake_item(2)}
        board = leaderboard(score_runs(runs, items))
        self.assertEqual(board.iloc[0]["api_fail"], 1)
        self.assertEqual(board.iloc[0]["split"], "val")
        self.assertEqual(board.iloc[0]["shots"], "train")
        self.assertEqual(board.iloc[0]["recall@0.5"], 0.5)


class TestRunner(unittest.TestCase):
    def test_dry_run_writes_nothing_and_counts(self):
        items = [fake_item(0), fake_item(1)]
        root = make_root(items)
        s = run_benchmark([mock_spec("empty")], root=root, dry_run=True, budget_usd=10)
        self.assertEqual(s.attempted, 2)
        self.assertGreater(s.spent, 0)
        self.assertEqual(load_runs(root), [])

    def test_mock_run_then_cache_hit(self):
        items = [fake_item(0)]
        root = make_root(items)
        s1 = run_benchmark([mock_spec("oracle")], root=root, budget_usd=10, log=lambda *a: None)
        self.assertEqual((s1.attempted, s1.ok), (1, 1))
        s2 = run_benchmark([mock_spec("oracle")], root=root, budget_usd=10, log=lambda *a: None)
        self.assertEqual((s2.attempted, s2.cached), (0, 1))
        run = load_runs(root)[0]
        self.assertEqual([l["price"] for l in run["levels"]], [5.5, 6.2])
        self.assertEqual(run["split"], "manifest")

    def test_budget_cap(self):
        items = [fake_item(i) for i in range(3)]
        root = make_root(items)
        s = run_benchmark([mock_spec("noisy")], root=root, dry_run=True, budget_usd=0.0001)
        self.assertEqual(s.skipped_budget, 3)

    def test_case_insensitive_items_and_no_match_message(self):
        root = tempfile.mkdtemp()
        logs = []
        s = run_benchmark([mock_spec("empty")], root=root, split="val", explanations_root=DATASET_ROOT,
                          only_items=["pton"], dry_run=True, budget_usd=10, log=logs.append)
        self.assertEqual(s.attempted, 1)
        logs.clear()
        s = run_benchmark([mock_spec("empty")], root=root, split="val", explanations_root=DATASET_ROOT,
                          only_items=["missing"], dry_run=True, budget_usd=10, log=logs.append)
        self.assertEqual(s.attempted, 0)
        self.assertIn("13_PTON_2024-08-26", logs[0])

    def test_split_fewshot_run_records_context(self):
        root = tempfile.mkdtemp()
        s = run_benchmark([mock_spec("oracle")], root=root, split="val", shot_splits=("train",),
                          explanations_root=DATASET_ROOT, only_items=["pton"], budget_usd=10,
                          log=lambda *a: None)
        self.assertEqual((s.attempted, s.ok), (1, 1))
        run = load_runs(root)[0]
        self.assertEqual(run["split"], "val")
        self.assertEqual(run["shots"], "train")
        self.assertEqual(run["n_shots"], 11)

    def test_split_and_test_shot_guards(self):
        with self.assertRaises(ValueError):
            run_benchmark([mock_spec("empty")], root=tempfile.mkdtemp(), split="val", shot_splits=("val",),
                          explanations_root=DATASET_ROOT, dry_run=True)
        with self.assertRaises(ValueError):
            run_benchmark([mock_spec("empty")], root=tempfile.mkdtemp(), split="test", shot_splits=("test",),
                          explanations_root=DATASET_ROOT, dry_run=True)

    def test_empty_and_truncated_responses_are_errors_with_cost(self):
        root = make_root([fake_item(0)])
        s = run_benchmark([mock_spec("blank"), mock_spec("truncated")], root=root, budget_usd=10,
                          log=lambda *a: None)
        self.assertEqual((s.ok, s.failed), (0, 2))
        runs = load_runs(root)
        self.assertEqual([r["status"] for r in runs], ["error", "error"])
        self.assertEqual(runs[0]["finish_reason"], "stop")
        self.assertEqual(runs[1]["finish_reason"], "length")
        self.assertTrue(all(r["cost_usd"] > 0 for r in runs))

    def test_key_stable(self):
        spec = mock_spec("empty")
        a = run_key(spec, "i", "rubric", 0, "p", PNG, 8000)
        self.assertEqual(a, run_key(spec, "i", "rubric", 0, "p", PNG, 8000))
        self.assertNotEqual(a, run_key(spec, "i", "rubric", 1, "p", PNG, 8000))
        self.assertNotEqual(a, run_key(spec, "i", "rubric", 0, "p", PNG, 8000, "shots"))

    def test_missing_keys(self):
        os.environ.pop("FAKE_ENV_KEY", None)
        from src.vlm.models import ModelSpec
        spec = ModelSpec(key="t", label="t", vendor="t", tier="mid", route="openrouter", api_model="x",
                         key_env="FAKE_ENV_KEY", openrouter_id="x", price_in=1, price_out=1,
                         est_reasoning_tokens=0, context="")
        self.assertEqual(missing_keys([spec]), {"FAKE_ENV_KEY": ["t"]})


class TestPackets(unittest.TestCase):
    def test_manifest_structure_if_built(self):
        if not os.path.exists(os.path.join("data", "vlm", "manifest.json")):
            self.skipTest("packets not built yet")
        mf = load_manifest("data/vlm")
        dataset_paths = {}
        for split in ("train", "val"):
            dataset_paths.update({i["id"]: i for i in load_split_items(split, DATASET_ROOT)})
        for it in mf["items"]:
            for key in ("id", "symbol", "asof", "chart_blind", "chart_labeled", "ohlcv", "gold_levels", "price", "atr14"):
                self.assertIn(key, it)
            packet = dataset_paths[it["id"]]
            for key, packet_key in (("chart_blind", "chart_path"), ("ohlcv", "ohlcv_path")):
                path = os.path.join("data", "vlm", it[key])
                self.assertTrue(os.path.exists(path) or os.path.exists(packet[packet_key]), it[key])
            self.assertTrue(it["chart_blind"].startswith("charts_without_labels"))
            self.assertTrue(it["chart_labeled"].startswith("charts_with_labels"))
            self.assertTrue(all(g > 0 for g in it["gold_levels"]))


if __name__ == "__main__":
    unittest.main()
