"""S1 (tested-partial): Hindi speech is detected and answered with the Hindi voice. Whisper-base Hindi
transcripts are too noisy for the Hindi cache to match reliably (see docs/DEVIATIONS.md)."""
import soundfile as sf


def test_hindi_routes_to_hindi_voice(ollama_up, make_wav, tmp_path):
    from pv.audio.playback import FilePlayer
    from pv.pipeline import Features, Pipeline
    from pv.runtime import Runtime
    from pv.tts.piper_tts import PiperTTS
    from pv.config import model

    hindi = next(model("tts").glob("hi_IN-*.onnx")).stem
    tts = PiperTTS(hindi, threads=2, length_scale=1.0)
    path = tmp_path / "hi.wav"
    sf.write(path, tts.synth("आप कैसे हैं", do_normalize=False), tts.sample_rate)

    rt = Runtime(ram_mb=8192)
    out = tmp_path / "reply.wav"
    p = Pipeline(rt, FilePlayer(out), Features.full(), mode="ptt")
    try:
        tl = p.run_wav(path)
        assert tl.lang == "hi", (tl.lang, tl.transcript)
        assert p.last_voice.startswith("hi_IN"), p.last_voice      # the reply was spoken with the Hindi voice
        assert out.exists() and tl.reply
        english = p.run_wav(make_wav("Do I need internet to talk to you?", "en"))
        assert english.lang == "en" and english.path == "cache"
    finally:
        p.close()
        rt.stop()
