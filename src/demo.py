# filepath: /streamlit-audio-recorder/src/demo.py

import io
import os
import streamlit as st
from st_audiorec import st_audiorec
from voice_spoof_detection import compute_hash, get_model, run_model, AudioProcessor
import numpy as np
import torchaudio
import torch

st.set_page_config(page_title="Detection System Voice Spoofing Attacks")

def audiorec_demo_app():
    st.title('Hệ thống phát hiện tấn công giả mạo giọng nói')
    st.markdown('Implemented by [VI&HA]')
    st.write('\n\n')

    # Thu âm trực tiếp
    st.subheader("🎤 Record Audio")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        temp_file_path = os.path.join("file_input", "recorded_audio.wav")
        with open(temp_file_path, "wb") as f:
            f.write(wav_audio_data)

        try:
            with st.spinner("Đang phân tích..."):
                # Đọc và xử lý file âm thanh
                waveform, sample_rate = torchaudio.load(temp_file_path)

                # Nếu waveform là stereo, chuyển sang mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Resample nếu cần
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

                # Chạy mô hình
                likelihood = run_model(waveform.numpy())

                # Hiển thị kết quả với cảnh báo
                if likelihood > 70:
                    st.error(f"🚨 **Khả năng là giọng giả:** {likelihood:.1f}% ❗")
                elif likelihood > 40:
                    st.warning(f"⚠️ **Khả năng là giọng giả:** {likelihood:.1f}% ⚠️")
                else:
                    st.success(f"✔️ **Khả năng là giọng giả:** {likelihood:.1f}% ✔️")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Tải lên file âm thanh
    st.subheader("📁 Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Chọn file .wav hoặc .flac",
        type=["wav", "flac"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Bạn chỉ được chọn tối đa 10 file.")
        else:
            results = []
            with st.spinner("Đang phân tích..."):
                # Xử lý từng file
                for uf in uploaded_files:
                    uf.seek(0)
                    name = uf.name
                    ext = name.rsplit(".", 1)[-1].lower()
                    wav, sample_rate = torchaudio.load(uf, format="flac" if ext == "flac" else None, backend="soundfile")

                    # Resample nếu cần
                    if sample_rate != 16000:
                        wav = torchaudio.functional.resample(wav, sample_rate, 16000)

                    # Chạy mô hình
                    prob = run_model(wav.numpy())
                    results.append((uf, name, prob))

            # Hiển thị kết quả phân tích
            st.subheader("Kết quả phân tích")
            for file_obj, name, prob in results:
                if prob > 45:
                    sym, col = "❗", "red"
                elif prob > 30:
                    sym, col = "⚠️", "orange"
                else:
                    sym, col = "✔️", "green"
                st.markdown(
                    f"- **{name}**: <span style='color:{col};'>{prob:.2f}% {sym}</span>",
                    unsafe_allow_html=True
                )
                with st.expander(f"🔊 Nghe lại {name}"):
                    st.audio(file_obj, format=f"audio/{ext}")

if __name__ == '__main__':
    audiorec_demo_app()