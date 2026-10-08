# PocketVoice

A CPU-only, fully offline voice assistant (ASR -> LLM -> TTS) built for hackathon problem statement HNX26EPS08.
It runs inside a **simulated edge device** (default 4 cores / 4096 MB, enforced by the OS) and is benchmarked against a
standard quantized stack on latency, footprint and energy. Spec: [SPEC.md](SPEC.md). Results: [reports/report.md](reports/report.md),
[reports/WRITEUP.md](reports/WRITEUP.md). Honest audit of every requirement: [docs/AUDIT.md](docs/AUDIT.md).

## One-command operation (Windows, PowerShell)

```powershell
scripts\setup.ps1     # once, online: Python 3.12 venv, packages, Ollama models, voices, caches, manifest (safe to re-run)
scripts\run.ps1       # offline: CPU-only Ollama on 127.0.0.1:11435 + limiter + app + dashboard on http://127.0.0.1:8765
scripts\demo.ps1      # same, and opens the dashboard
scripts\bench.ps1     # baseline, ablation ladder, quantization sweep, degradation test, cache + quality eval, reports
```

Default is **hands-free**: just talk. It listens continuously, answers **only when you speak**, and stays silent otherwise (silence,
fan noise and its own voice are ignored). No SPACE and no wake word needed. Other modes: `scripts\run.ps1 -Mode wake` ("Hey Jarvis" or
SPACE) and `-Mode ptt` (SPACE only). While running: **T** tightens the device to 2 cores / 1536 MB live, **R** restores, **Q** quits.
It does not listen while it is speaking (so it never answers itself); pass `-Extra "--barge-in"` with headphones to interrupt it.
For better answers on a roomier device: `scripts\run.ps1 -Profile roomy` (3B model, 6 cores / 6 GB). Checklist: [docs/LIVE_TEST.md](docs/LIVE_TEST.md).

Loopback without a microphone (what the tests use):

```powershell
.\.venv\Scripts\python.exe -m pv.app --wav bench\testset\o03.wav --no-audio-out --out data\reply.wav --no-dash
.\.venv\Scripts\python.exe -m pv.baseline --wav bench\testset\o03.wav --no-audio-out
```

## How it works

```
mic -> Silero VAD + openWakeWord -> Moonshine streaming ASR -> router -> sentence chunker -> Piper (2-slot) -> speaker
                                          skills -> semantic cache (pre-rendered audio) -> Ollama LLM (CPU)
                  governor: tiers T0..T4 driven by RSS / CPU / tokens-per-second / latency; guards: gpucheck, netguard, limiter
```

* **R2 offline**: `pv/guard/netguard.py` blocks and counts every non-loopback socket; models load from `models/` only.
* **R3 CPU-only**: no torch, ORT CPU provider only (asserted), `num_gpu=0`, Ollama started with CUDA/HIP/Vulkan hidden.
* **R4 device limit**: CPU affinity (LLM and audio on separate cores when >= 4) + Windows Job Object memory cap over the app *and* the Ollama tree; `tighten()` works live.
* Novel parts: compute-avoidance router, adaptive endpointing, wake-time prefix warming, tier governor, barge-in, sentence-level two-slot streaming, speakable-text normaliser.

## Layout

`pv/` runtime · `config/` default + profiles (`eco`, `pi_like`, `roomy`; use `scripts\run.ps1 -Profile pi_like`) · `bench/` test-set generator,
benchmarks, report · `dash/` dashboard · `tests/` pytest · `docs/DEVIATIONS.md` every substitution from the spec.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q      # needs models; starts the CPU-only Ollama itself
```
