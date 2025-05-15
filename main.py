import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="IntisariMi", layout="wide")

# Load model dan tokenizer (jalankan sekali saat pertama load)
@st.cache_resource
def load_model():
    model_path = "annisareida04/TextSummarizationBartBaseDialogsum"  #gantidengan path model kamu
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Fungsi untuk meringkas
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

# UI Streamlit


st.title("📝 Text Summarization")

with st.expander("📌 Instructions"):
    st.markdown("""
    1. Masukkan teks panjang (misalnya artikel atau berita) pada kolom di bawah.
    2. Klik tombol **Ringkas Teks**.
    3. Hasil ringkasan akan muncul di bawahnya.
    """)

text_input = st.text_area("Masukkan teks untuk diringkas:", height=300)

if st.button("Ringkas Teks"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        with st.spinner("Sedang meringkas..."):
            result = summarize(text_input)
            st.subheader("Hasil Ringkasan:")
            st.success(result)

st.markdown("""
    <div style="text-align: left; font-size: 12px; color: gray;">
        Dibuat untuk memenuhi proyek tugas akhir mata kuliah NLP, oleh:
        <ul style="font-size: 8px; color: gray;">
            <li>Annisa Reida Raheima</li>
            <li>Revalina Ramadhani</li>
            <li>Zweta Anggun Syafara</li>
            <li>Nabila Kurnia Aprianti</li>
            <li>Evan Febrian</li>
        </ul>
        Jurusan Teknik Informatika<br>
        Fakultas Ilmu Komputer Universitas Sriwijaya 2025
    </div>
    """, unsafe_allow_html=True)
