# Self-audit (strict)

Machine: AMD Ryzen 5 5600H (6 cores / 12 threads), 15.7 GB RAM, Windows 11. Declared limit: **4 cores / 4096 MB**.
All numbers below come from `bench/out/*.json`, `reports/proofs.json` or the pytest/CLI output quoted in the "Evidence" column.
PASS = verified by a command that was run; PARTIAL = works but misses part of the requirement or evidence is reduced; FAIL = not met.

## Requirements

| # | Requirement | Status | Command run | Evidence |
|---|---|---|---|---|
| R1 | Full ASR -> LLM -> TTS loop on-device | **PASS** | `pytest tests/test_e2e.py` | 3 tests pass: baseline loop; streaming pipeline through all 3 routes (skill/cache/llm) with every timeline stamp populated and a reply WAV written; wake word ("Hey Jarvis" + question, no push-to-talk) -> "That is 391." |
| R2 | Fully offline | **PASS** (software proof) | `pytest tests/test_offline.py`; `python scripts/proofs.py` | External connect/DNS/HTTP raise `OfflineViolation` and are counted; loopback allowed. `proofs.json`: network attempts during a real run = 0, deliberate probe to 8.8.8.8 blocked. **Not done by me:** a physical Wi-Fi-off run (see `docs/LIVE_TEST.md` section 3); optional `scripts/offline_fw.ps1` untested (needs admin). |
| R3 | CPU-only (gate) | **PASS** | `pytest tests/test_gpu_free.py`; `python scripts/proofs.py` | All ORT sessions `["CPUExecutionProvider"]`; no torch/tensorflow/onnxruntime-gpu in `pip list`; Ollama `/api/ps` `size_vram: 0` for `qwen2.5:1.5b-instruct`; log line `inference compute id=cpu library=cpu`. **Not done:** Task Manager GPU-graph screenshot (needs a person). |
| R4 | Simulated edge device enforced by OS | **PASS** | `pytest tests/test_limits.py`; `proofs.py` | Affinity applied (2 cores -> `[0, 2]`); Job Object memory cap: a child allocating 100 MB under a 400 MB cap succeeds, 900 MB raises `MemoryError`. Ollama tree is pinned and joined to the job by the limiter. Live `tighten()` works (see S4/D-13). |
| R5 | Wake word / VAD | **PASS** | `pytest tests/test_wake_vad.py` | VAD voiced fraction >0.7 on speech, 0.0 on silence, <0.1 on noise; 0.16 ms/frame (target <1 ms); wake score >= threshold on "Hey Jarvis", below on other speech and silence. Wake model tested on synthetic speech only. |
| R6 | Beat baseline latency | **PASS** | `python -m bench.run_ablation` (+ `A7` re-run) | p50 **4,180 -> 1,205 ms (-71%)**, p95 **13,976 -> 2,121 ms (-85%)**, mean 5,158 -> 1200 ms; 180 baseline turns vs 120 turns for A7 (60 requests x 3 vs x 2 repeats). Synthetic speech, one machine. |
| R7 | Lower footprint (CPU, RAM, energy) | **PARTIAL** | same | Peak RSS 1,958 -> **1,733 MB (-11.5%)** (A7 after leak fix); CPU-s/turn 17.1 -> **5.0 (-71%)**. Energy: RAPL gross J/turn 121.9 -> 79.5 (-35%) but **net-of-idle 32.2 -> 35.1 J (no improvement; RAPL is whole-machine and noisy)**; CPU-seconds proxy 102.6 -> 29.9 J (assumption, 6 W/core). **Idle footprint is not lower**: idle CPU 1.95% -> 2.73%, idle RSS 1,502 -> 1,589 MB (extra always-on models: wake word, VAD, router). |
| R8 | Report CPU, RAM, energy | **PASS** | `python -m bench.make_report` | `reports/report.md` tables with CPU-s/turn, peak RSS and commit, J/turn (RAPL measured and proxy, labelled), idle CPU/RSS. |
| R9 | Short write-up | **PASS** | `bench/make_report.py` | `reports/WRITEUP.md`, ~700 words (under 2 pages), generated from the JSON. |
| S1 | Multilingual | **PARTIAL** | `pytest tests/test_multilingual.py` | Language detection (Whisper multilingual, parallel with speech), Hindi voice and Hindi prompt work; **Whisper-base Hindi transcripts are poor** so the Hindi cache rarely matches (D-12). Not claimed as reliable. |
| S2 | Smaller device (pi_like) | **PARTIAL** | `run_bench --system FULL --profile pi_like --cores 2 --ram 1536` | 20/20 completed within the cap (peak commit 1,223 MB), p50 578 ms; but the governor chose **T4 Survival** (the 0.5B model does not fit under 1.5 GB), so open questions get the fallback reply. Only 20 skill-type requests were run. No real-device test. |
| S3 | Cached TTS | **PASS** | `bench.run_bench` A7; `bench.cache_eval` | Cache route p50 **534 ms** (22 of 120 turns; pre-rendered audio, no synthesis). Hit rate and error rate: below. |
| S4 | Quality as limit tightens | **PARTIAL** | `bench.run_degrade --n 20 --steps 4:4096,2:2048,2:1536` | Governor: 100% completion at all 3 limits, quality 0.95 -> 0.70 -> 0.70, p50 1,090 -> 618 -> 605 ms. Fixed Full config: 100% at 4c/4GB, **fails to start (HTTP 500) at 2c/2GB and 2c/1.5GB**. Reduced scope: 3 limits x 20 requests (spec: 5 x 40); per-tier quality run not done. |

## Rubric

| Item | Status | Evidence / caveat |
|---|---|---|
| Latency vs baseline (25%) | **PASS** | R6. Per-route p50 (A7): skill 741 ms, cache 534 ms, LLM 1,509 ms (p95 2,228). Spec targets were skill <=400, cache <=450, LLM <=1,500/2,500 ms: **skill and cache targets missed** (the measured span starts at the last voiced frame, so ~340 ms mean endpoint hangover is inside every number), LLM target met. |
| Footprint vs baseline (25%) | **PARTIAL** | R7. CPU and peak RAM clearly lower; energy not demonstrated; idle not lower. |
| Quantization + offload (15%) | **PASS** (with caveat) | `python -m bench.run_quant_sweep`: Q8_0/Q5_K_M/Q4_K_M/Q3_K_M 1.5B and 0.5B Q4: 24.7/35.6/39.0/42.6/80.1 tok/s, RSS 1,657/1,156/1,028/874/451 MB; KV f16 vs q8_0 and flash attention on/off (RSS within 1.5%, speed within 2%); ASR int8 vs fp32 and Moonshine tiny vs small (WER, RTF, RSS); TTS medium vs low (RTF 0.07/0.05); load 1.67 s / unload 0.01 s; core pinning 39.7 vs 39.0 tok/s. Caveat: the 20-prompt keyword rubric gives 1.00 for every 1.5B variant, so it cannot separate them; TTS subjective rating needs humans. |
| Graceful degradation (15%) | **PASS** (reduced) | S4. Governor unit tests (9) + live-tighten integration test pass. |
| Research contribution + ablation (20%) | **PASS** | Ladder A0-A7 complete (below). Caveats: A1-A7 used 2 repeats (baseline 3, D-09); A1-A6 predate the Piper-leak fix (D-15); one overlapping test run touched an earlier A1 attempt, so A1 was re-run clean. |
| Bonus: multilingual / smaller device / cached TTS | **PARTIAL** | S1 partial, S2 partial, S3 pass. |

## Ablation ladder (60 requests; p50 / p95 ms; RSS MB; CPU-s/turn; quality)

| Step | p50 | p95 | RSS | CPU-s | quality |
|---|---|---|---|---|---|
| A0 baseline | 4,180 | 13,976 | 1,958 | 17.1 | 0.75 |
| A1 +Moonshine streaming | 5,033 | 11,859 | 2,759 | 11.0 | 0.64 |
| A2 +sentence-level TTS | 4,144 | 6,634 | 2,295 | 11.3 | 0.70 |
| A3 +voice prompt/caps | 2,784 | 3,765 | 1,978 | 6.6 | 0.64 |
| A4 +prefix warming | 1,608 | 2,492 | 1,937 | 6.8 | 0.66 |
| A5 +adaptive endpointing | 1,538 | 2,341 | 1,954 | 6.7 | 0.68 |
| A6 +router (skills+cache) | 1,154 | 1,956 | 2,028 | 4.9 | 0.93 |
| A7 +governor (post-fix re-run) | 1,205 | 2,121 | 1,733 | 5.0 | 0.93 |

Notes: A1 is *slower* at p50 than the baseline (full uncapped replies, split cores); the gains come from A2 (streaming) onward. The A3->A4 drop (-42%) is large; I did not separate "prefix warming" from "keeps the model resident/KV hot", so I do not claim the mechanism. Quality jumps at A6 because skills answer time/date/maths/units deterministically; the keyword rubric is lenient (a generic fallback reply can match loose keywords), so treat quality as indicative only.

## Cache and router honesty

Chosen on the dev paraphrases (tau 0.75, margin 0.03): dev hit 60%, wrong 8% of hits, false hits on out-of-domain 10%. **Held-out (n=10, authored by me, not independent): hit 50%, wrong 20% of hits (1 of 5)**, false hits 0%. At the spec default (0.80/0.05) held-out hit is only 20%. The router answered 45% of ladder turns without the LLM. Known failure: topic questions match assistant-directed intents ("why do we sleep" -> "do you sleep").

## Re-run results (final)

```
pytest tests -q                       -> 59 passed in 145.68s
python scripts/proofs.py              -> PROOFS PASS (no banned packages; ORT CPU only; size_vram 0; 0 network attempts, 8.8.8.8 probe blocked; 400 MB cap: 100 MB ok, 900 MB MemoryError)
run_bench --system FULL (multilingual) -> 20/20 completed, p50 1,121 ms, peak RSS 1,924 MB, peak commit 2,571 MB (cap 4,096)
```

## Definition of done

- [x] Voice in -> voice out, offline, CPU-only, under the limit, with wake word/VAD (synthetic speech).
- [x] GPU proof and network proof on the dashboard (verified via `/events`, shown in the proof panel).
- [~] Baseline vs ours: latency and CPU/RAM lower; energy and idle footprint not shown lower (R7).
- [~] Ablation, quantization sweep, degradation generated; degradation reduced; per-tier quality not run.
- [x] Router/cache savings and error rate reported.
- [x] Live limit tightening without a crash (`test_live_tighten...`, after fixing D-13).
- [ ] **Backup demo video: not made** (needs a person). Short write-up exists.
- [x] `docs/DEVIATIONS.md` lists substitutions (D-01 .. D-15).

## Not verified by me (needs a person)

Real microphone/speaker round trip and barge-in through speakers; Wi-Fi-off run; Task Manager GPU graph; human TTS/answer ratings; real-voice recordings; teammate-written held-out questions; the optional firewall script.

## Known issues

- Skill and cache routes miss the spec's 400/450 ms targets (741/534 ms p50).
- Hindi transcription is weak (D-12). Cache coverage of unseen paraphrases is about half.
- Windows quirk: Ollama joined to the limiter's Job Object exits with the app that created the job (by design in `run.ps1`; tests re-launch it per test).
