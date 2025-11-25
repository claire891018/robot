from transformers import pipeline
import numpy as np
import wave

def save_wav(path: str, audio, sr: int):
    # 轉成 numpy，壓到 -1~1 再轉 int16
    audio = np.asarray(audio, dtype=np.float32)

    # 如果是多維，全部攤平成一條
    if audio.ndim > 1:
        audio = audio.reshape(-1)

    max_val = np.max(np.abs(audio)) + 1e-8
    audio = audio / max_val
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)       # Bark 預設單聲道就先寫 1
        wf.setsampwidth(2)       # 16 位元 = 2 bytes
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

if __name__ == "__main__":
    # 有 GPU 可以加 device="cuda"
    pipe = pipeline("text-to-speech", model="suno/bark-small")

    text = "最近天氣好冷，快凍死！"
    out = pipe(text)

    audio = out["audio"]
    sr = out["sampling_rate"]

    save_wav("bark_zh.wav", audio, sr)
    print("已輸出 bark_zh.wav")

# python single_inference.py \
#     --speaker_prompt_audio_path "data/TTS_voice.wav" \
#     --speaker_prompt_text_transcription "" \
#     --content_to_synthesize "最近天氣好冷，快凍死！" \
#     --output_path results/out.wav 2>/dev/null