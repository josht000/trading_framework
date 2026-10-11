| model | key | vendor | tier | in_$/M | out_$/M | est_in_tok | est_out_tok | est_$/chart | est_$/run(19x1) | context | open | arena_vision | charxiv_r | chartography | mmmu_pro | vision_note | route | first_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi K3 | kimi-k3 | Moonshot | mid | 0.720 | 15.0 | 6605 | 3250 | 0.054 | 1.017 | 1M | yes |  | 91.3 |  |  | CharXiv-R 91.3 with tools / 84.8 no tools; open weights | openrouter |  |
| Muse Spark 1.3 | muse-spark-1.3 | Meta | mid | 1.250 | 4.250 | 6605 | 2250 | 0.018 | 0.339 | 1M | no | 1310.0 | 88.4 |  |  | Arena Vision #11 (1310); CharXiv-R 88.4 is Muse Spark 1.1 | openrouter |  |
| GPT-6.1 Sol | gpt-6.1-sol | OpenAI | mid | 2.000 | 10.0 | 6437 | 2750 | 0.040 | 0.767 | 1.05M | no | 1291.0 |  | 53.6 | 83.0 | Arena Vision 1291 (gpt-6.1-sol-max, preliminary); Chartography 53.6 is GPT-6 Sol (Anthropic-reported); MMMU-Pro 83.0 is GPT-5.6 Sol | openrouter | yes |
| Qwen3.8 Max | qwen3.8-max | Alibaba | mid | 2.000 | 6.000 | 6605 | 2750 | 0.030 | 0.564 | 1M | no | 1314.0 |  |  |  | Arena Vision #8 (1314); proprietary | openrouter | yes |
| Gemini 3.1 Pro (preview) | gemini-3.1-pro | Google | mid | 2.000 | 12.0 | 5637 | 3250 | 0.050 | 0.955 | 1M | no | 1305.0 | 83.2 |  |  | CharXiv-R 83.2 with tools; Arena 1305 is listed as gemini-3-pro | openrouter |  |
| Grok 4.7 | grok-4.7 | xAI | mid | 2.000 | 6.000 | 6605 | 2250 | 0.027 | 0.507 | 500K | no | 1282.0 |  |  |  | Arena 1282 is grok-4.5; no 4.7 vision numbers found | openrouter |  |
| Claude Opus 5.5 | claude-opus-5.5 | Anthropic | mid | 4.000 | 20.0 | 7960 | 2750 | 0.087 | 1.650 | 1M | no | 1321.0 |  | 64.4 | 87.7 | Anthropic's best Opus for vision; Chartography 64.4 no tools / 89.0 with tools; AA MMMU-Pro 87.7; Arena 1321 is Opus 5 (5.5 not listed yet) | openrouter |  |
| Claude Haiku 5.5 | claude-haiku-5.5 | Anthropic | budget | 0.100 | 0.500 | 7960 | 1750 | 0.002 | 0.032 | 1M | no |  |  | 46.4 |  | Chartography 46.4% no tools (Anthropic) | openrouter | yes |
| GPT-6 Luna | gpt-6-luna | OpenAI | budget | 0.100 | 0.500 | 6437 | 1050 | 0.001 | 0.022 | 1.05M | no |  |  | 29.1 |  | Chartography 29.1% no tools (reported by Anthropic) | openrouter | yes |
| Qwen3.8 Flash | qwen3.8-flash | Alibaba | budget | 0.150 | 0.470 | 6605 | 1750 | 0.002 | 0.034 | 1M | no |  | 90.6 |  |  | CharXiv-R 90.6 (LLMBoard; tool setting not stated) | openrouter | yes |
| GLM-5.3 Flash | glm-5.3-flash | Zhipu | budget | 0.150 | 0.500 | 6605 | 1750 | 0.002 | 0.035 | 1M | yes | 1301.0 | 89.4 |  |  | Arena 1301; CharXiv-R 89.4 with tools; open weights | openrouter |  |
| DeepSeek V4 Flash Vision Exp | deepseek-v4-flash-vision-exp | DeepSeek | budget | 0.216 | 0.647 | 6605 | 3250 | 0.004 | 0.067 | 1.05M | no |  |  |  |  | Image input confirmed; explicitly experimental; no chart-specific benchmark located | openrouter | yes |
| Gemini 3.5 Flash-Lite | gemini-3.5-flash-lite | Google | budget | 0.300 | 2.500 | 5637 | 1050 | 0.004 | 0.082 | 1M | no |  |  |  |  | no public vision benchmark found | openrouter |  |
| DeepSeek V4.1 Flash | deepseek-v4.1-flash | DeepSeek | budget | 0.300 | 1.200 | 6605 | 3250 | 0.006 | 0.112 | 1.05M | no |  |  |  |  | Native image input confirmed; no chart-specific benchmark located | openrouter | yes |
| Gemini 3.8 Flash | gemini-3.8-flash | Google | budget | 0.750 | 3.750 | 5637 | 2750 | 0.015 | 0.276 | 1M | no | 1316.0 | 86.2 |  |  | Arena Vision #5 (preliminary); CharXiv-R 86.2 no-tools, Google self-reported | openrouter | yes |
| Qwen2.5 VL 72B | qwen2.5-vl-72b | Alibaba | budget | 0.800 | 1.000 | 6605 | 250 | 0.006 | 0.105 | 128K | yes |  |  |  |  | Open-weights (Apache 2.0) 2025 vision model; no chart-specific scores on file | openrouter |  |

- Prices are USD per 1M tokens (standard, non-batch, short-context) from the vendors' official pricing pages checked 2026-10-08; models reached only through OpenRouter use OpenRouter's live list price.
- Vision signals are proxies only. No public benchmark scores support/resistance reading on price charts. Numbers come from different sources and settings (tools vs no tools, vendor self-reported vs leaderboard) and are not comparable across rows.
- est_reasoning_tokens is an ASSUMED hidden-reasoning output size per request; actual usage is recorded on every real run and replaces these estimates.
- Gemini 3.8 Flash price is introductory ($0.75/$3.75) through 2026-12-31, then $1.50/$7.50.
- DeepSeek V4.1 Flash and V4 Flash Vision Exp IDs, image support, prices, and contexts verified on OpenRouter 2026-10-09. Vision Exp is explicitly experimental; neither has chart-specific performance data in this registry.
- DeepSeek standard rates are OpenRouter list prices as of 2026-10-09; cache-read discounts are not used in estimates.
- DeepSeek V4.1 Flash / V4 Flash Vision Exp are reasoning models: in a first unconstrained run (8000-token cap) V4.1 Flash returned 8/8 empty answers and Vision Exp 2/8, with ~4800-8000 hidden reasoning tokens each. They now request reasoning effort 'low' (est. 3000 reasoning tokens, unverified until re-run) and runs use --max-tokens 16000.
