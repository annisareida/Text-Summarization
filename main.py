import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="📜 IntisariKu", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model_path = "annisareida04/TextSummarizationBartBaseDialogsum"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Fungsi meringkas
def summarize(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# CSS Styling Warna & Tampilan
st.markdown("""
    <style>
    body {
        background-color: #eef6fc;
    }
    .main {
        background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #5d6d7e;
        margin-bottom: 25px;
    }
    .summary-box {
        background-color: #ffffff;
        border-left: 5px solid #3498db;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        font-size: 16px;
        line-height: 1.6;
    }
    .footer {
        text-align: left;
        font-size: 13px;
        color: #7f8c8d;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        padding: 0.5em 2em;
        border-radius: 8px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📜 IntisariKu</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplikasi Ringkasan Teks Otomatis Berbasis NLP</div>', unsafe_allow_html=True)

# Petunjuk
with st.expander("📌 Cara Menggunakan"):
    st.markdown("""
    1. Masukkan teks panjang (misalnya artikel atau berita).
    2. Klik tombol **Ringkas Teks**.
    3. Hasil ringkasan akan tampil di bawah.
    """)

# Input
text_input = st.text_area("📝 Masukkan teks panjang di sini:", height=300, placeholder="Contoh: Pemerintah mengumumkan kampanye vaksinasi nasional...")

# Button & Output
if st.button("🔍 Ringkas Teks"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        with st.spinner("🤖 Model sedang meringkas teks..."):
            result = summarize(text_input)
            st.markdown(f'<div class="summary-box"><b>📄 Hasil Ringkasan:</b><br><br>{result}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Dibuat untuk memenuhi proyek tugas akhir mata kuliah NLP oleh:
        <ul>
            <li>Annisa Reida Raheima</li>
            <li>Revalina Ramadhani</li>
            <li>Zweta Anggun Syafara</li>
            <li>Nabila Kurnia Aprianti</li>
            <li>Evan Febrian</li>
        </ul>
        Jurusan Teknik Informatika<br>
        Fakultas Ilmu Komputer, Universitas Sriwijaya – 2025
    </div>
""", unsafe_allow_html=True)
