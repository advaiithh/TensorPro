# PROGRESS

## C0 Setup: DONE
- Python 3.12.10 venv; all SPEC 3.3 packages installed (no torch/cuda: `pip list` grep is empty).
- Ollama pulled: qwen2.5:1.5b-instruct, qwen2.5:0.5b-instruct. Own CPU-only server on 127.0.0.1:11435.
- Verified: 1.5B streams at ~37 tok/s on CPU and `/api/ps` reports `size_vram: 0`.
- `scripts/setup_models.py` + `scripts/verify_assets.py`: 102 files / 848 MB, `models/MANIFEST.json` written.
- Each model ran standalone: Piper (RTF 0.09), Silero VAD (77/85 voiced frames on speech, 0.014 on silence),
  Whisper base.en (823 ms for a 2.7 s utterance), Moonshine tiny/small (exact transcript, finalize 80/122 ms).

## C1 Baseline loop: DONE
- `python -m pv.baseline --wav ... --no-audio-out` works; timeline logged to `data/timelines/baseline.jsonl`.
- `tests/test_e2e.py` passes with the network tripwire on (0 external attempts).

## C2 Guards + metrics + harness: DONE
- gpucheck / netguard / limiter, resource + RAPL energy meters, 90-file synthetic test set, `bench/run_bench.py`.
- 14 guard tests pass; cap enforcement proven on a child process (`MemoryError` above the Job Object cap).
- Baseline: p50 4,180 ms, p95 13,976 ms, 180/180 completed, peak RSS 1,958 MB, 17.1 CPU-s/turn.

## C3-C6 Streaming core, router, governor, dashboard: DONE (tested)
- Moonshine streaming + Silero VAD + openWakeWord, sentence chunker + two-slot Piper, router (skills/cache/LLM),
  adaptive endpointing, prefix warming, governor T0-T4, barge-in, live dashboard, notes/timers, Hindi (tested-partial).
- Bugs found and fixed by measurement: Moonshine 25x slow when pinned (single-thread env), Piper memory leak
  (ORT arena), `run_wav` race (60 s waits), live `tighten()` crashing the app (Job cap below usage).

## C7 Evaluation: DONE (reduced scope, see AUDIT/DEVIATIONS)
- Ladder A0-A7, reduced degradation test (3 limits x 20 requests), quantization sweep, cache eval, pi_like check, report + write-up.
- Not run: per-tier quality eval, 5-limit x 40-request degradation.

## C8 Polish: DONE except human-only items
- README, WRITEUP, AUDIT, LIVE_TEST. Final: 59 tests pass; `scripts/proofs.py` PASS.
- Open: backup demo video, real-mic test, Wi-Fi-off run, human ratings.
