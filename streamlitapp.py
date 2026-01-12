import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Mangga",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS CUSTOM (PERBAIKAN UI & FUNGSI)
# ==========================================
st.markdown("""
<style>
    /* 1. GLOBAL RESET */
    .stApp {
        background-color: #F0FDF4 !important;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* 2. KARTU PUTIH UTAMA (WRAPPER) */
    .css-card {
        background-color: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #DCFCE7;
        margin-bottom: 20px;
        position: relative; /* Penting untuk overlay */
        padding-bottom: 20px;
    }

    /* 3. VISUAL KOTAK UPLOAD (GARIS PUTUS-PUTUS) */
    .upload-visual-box {
        border: 2px dashed #16A34A; /* Hijau */
        border-radius: 12px;
        padding: 40px 20px;
        text-align: center;
        background-color: #F8FDF9;
        margin-top: 20px;
        cursor: pointer;
        transition: 0.3s;
        min-height: 200px; /* Tinggi minimum agar pas dengan uploader */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .upload-visual-box:hover {
        background-color: #DCFCE7;
        border-color: #15803D;
    }

    /* 4. TRIK CSS: MEMBUAT UPLOADER DI AREA UTAMA MENJADI OVERLAY */
    /* Target hanya uploader di kolom utama (bukan sidebar) */
    section.main [data-testid="stFileUploader"] {
        position: relative;
        z-index: 99;
        opacity: 0; /* Transparan */
        margin-top: -225px; /* Tarik ke atas menutupi kotak visual */
        height: 220px; /* Samakan tinggi dengan kotak visual */
        cursor: pointer;
    }
    
    /* Hapus instruksi default agar area klik bersih */
    section.main [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
    section.main [data-testid="stFileUploaderDropzone"] {
        height: 100%;
        min-height: 100%;
        border: none;
    }

    /* 5. SIDEBAR UPLOADER (NORMALISASI) */
    /* Pastikan uploader di sidebar TETAP TERLIHAT dan BERFUNGSI normal */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        opacity: 1 !important;
        margin-top: 0 !important;
        height: auto !important;
        position: static !important;
    }

    /* 6. STYLE ELEMEN LAIN */
    h1, h2, h3, h4, h5, p, span, div, label { 
        color: #14532D !important; 
    }
    
    .result-bar {
        background: linear-gradient(90deg, #16A34A 0%, #059669 100%);
        color: white !important;
        padding: 15px; border-radius: 12px; font-size: 18px; font-weight: 600;
        margin-bottom: 20px;
    }
    
    button[kind="secondary"] {
        border: 1px solid #FECACA !important;
        color: #DC2626 !important;
        background-color: #FEF2F2 !important;
    }
    button[kind="primary"] {
        background-color: #059669 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOGIKA & DATA
# ==========================================
DISEASE_KB = {
    'Healthy': {
        'severity': 'Ringan',
        'desc': "Daun mangga tampak sehat dengan warna hijau segar. Tidak ada tanda penyakit.",
        'treatments': [
            {"title": "Penyiraman", "desc": "Siram teratur saat musim kemarau.", "icon": "💧"},
            {"title": "Sinar Matahari", "desc": "Pastikan sinar matahari penuh.", "icon": "☀️"},
            {"title": "Pemupukan", "desc": "Gunakan pupuk NPK seimbang.", "icon": "✨"}
        ]
    },
    'Anthracnose': {
        'severity': 'Berat',
        'desc': "Bercak hitam/cokelat gelap yang menyebar. Disebabkan jamur Colletotrichum.",
        'treatments': [
            {"title": "Sanitasi", "desc": "Potong dan bakar bagian terinfeksi.", "icon": "✂️"},
            {"title": "Fungisida", "desc": "Semprot fungisida Tembaga/Mankozeb.", "icon": "🛡️"},
            {"title": "Sirkulasi", "desc": "Kurangi kelembaban dengan memangkas.", "icon": "💨"}
        ]
    },
    'Bacterial Canker': {
        'severity': 'Sedang',
        'desc': "Bercak hitam berair yang sedikit menonjol atau pecah-pecah.",
        'treatments': [
            {"title": "Hindari Air Atas", "desc": "Jangan siram daun dari atas.", "icon": "🚫"},
            {"title": "Bakterisida", "desc": "Gunakan Copper Hydroxide.", "icon": "🛡️"},
            {"title": "Sterilisasi", "desc": "Bersihkan alat potong dengan alkohol.", "icon": "🔧"}
        ]
    },
    'Powdery Mildew': {
        'severity': 'Sedang',
        'desc': "Lapisan serbuk putih seperti tepung pada daun muda.",
        'treatments': [
            {"title": "Fungisida Sulfur", "desc": "Semprot Sulfur atau Minyak Neem.", "icon": "💦"},
            {"title": "Pangkas", "desc": "Kurangi kerimbunan pohon.", "icon": "☀️"},
            {"title": "Waktu", "desc": "Semprot pagi hari saat tenang.", "icon": "⏰"}
        ]
    }
}
CLASS_NAMES = list(DISEASE_KB.keys())

# Import Preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_efficientnet

@st.cache_resource
def load_model_from_path(path):
    return tf.keras.models.load_model(path)

def predict_img(image, model, m_type):
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img)
    if arr.shape[-1]==4: arr=arr[...,:3]
    arr = np.expand_dims(arr, axis=0)
    
    if m_type == "ResNet50": proc = pp_resnet(arr)
    elif m_type == "MobileNetV2": proc = pp_mobilenet(arr)
    elif m_type == "EfficientNetB0": proc = pp_efficientnet(arr)
    else: proc = arr/255.0
    return model.predict(proc)

# ==========================================
# 4. SIDEBAR (CONFIG)
# ==========================================
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Pilihan Arsitektur
    m_type = st.selectbox("Arsitektur", ["ResNet50", "MobileNetV2", "EfficientNetB0"])
    
    # --- LOGIKA BARU ---
    # Kita gunakan 'key' yang dinamis berdasarkan m_type.
    # Jika m_type berubah (misal dari ResNet ke MobileNet), key akan berubah,
    # sehingga uploader akan otomatis mereset (menjadi kosong) untuk arsitektur baru tersebut.
    # File lama pada arsitektur sebelumnya tersimpan di memori tapi tidak aktif.
    dynamic_key = f"uploader_{m_type}"
    
    # CSS di atas memastikan uploader Sidebar TETAP TERLIHAT
    up_model = st.file_uploader(f"Upload Model {m_type} (.h5)", type="h5", key=dynamic_key)
    
    model = None
    if up_model:
        # Simpan file sementara
        temp_path = f"temp_{m_type}.h5"
        with open(temp_path, "wb") as f:
            f.write(up_model.getbuffer())
            
        try:
            # Load model
            model = load_model_from_path(temp_path)
            
            # Tampilkan pesan sukses dalam kotak hijau (mirip screenshot)
            st.markdown(f"""
            <div style="background-color: #DCFCE7; border: 1px solid #16A34A; padding: 10px; border-radius: 8px; color: #14532D; display: flex; align-items: center; gap: 8px; margin-top: 10px;">
                <span style="font-size: 18px;">✅</span>
                <span style="font-weight: 600;">{m_type} Siap!</span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            
    else:
        # Jika belum ada file upload untuk arsitektur yang dipilih
        st.info(f"Silakan upload file .h5 untuk **{m_type}**.")

# ==========================================
# 5. UI UTAMA (HEADER)
# ==========================================
status_html = f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="background: linear-gradient(135deg, #16A34A, #15803D); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px;">🍃</div>
        <div>
            <div style="font-weight: bold; color: #14532D; font-size: 18px;">Deteksi Penyakit Daun Mangga</div>
            <div style="font-size: 12px; color: #16A34A;">Sistem diagnosa cerdas untuk kesehatan tanaman mangga</div>
        </div>
    </div>
    <div style="font-size: 12px; font-weight: bold; color: {'#16A34A' if model else '#DC2626'}; background: white; padding: 5px 10px; border-radius: 10px; border: 1px solid {'#16A34A' if model else '#DC2626'};">
        {'✅ Sistem Siap' if model else '⚠️ Model Belum Diupload'}
    </div>
</div>
"""
st.markdown(status_html, unsafe_allow_html=True)

if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

col_left, col_right = st.columns([1, 1.2], gap="large")

# ==========================================
# 6. KOLOM KIRI (UPLOAD & PREVIEW)
# ==========================================
with col_left:
    # UPDATED: Desain disesuaikan persis dengan screenshot target (Bahasa Indonesia & Deskripsi Baru)
    # PENTING: Jangan tambahkan spasi di depan tag HTML di bawah ini.
    st.markdown("""
<div class="css-card">
<h3 style="color:#14532D; margin:0 0 10px 0; font-size:20px; font-weight:600;">Unggah Gambar Daun Mangga</h3>
<p style="color:#15803D; font-size:15px; margin-bottom:25px; line-height:1.6;">
Unggah foto daun mangga yang jelas untuk dianalisis. Sistem kami akan mendeteksi penyakit dan memberikan rekomendasi perawatan khusus untuk tanaman mangga Anda.
</p>
""", unsafe_allow_html=True)

    # --- UPLOADER NYATA (TRANSPARAN) ---
    # CSS akan menarik elemen ini ke atas (-225px) menutupi visual box di atas
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key=f"main_up_{st.session_state.uploader_key}", label_visibility="collapsed")

    # Jika file sudah diupload, tampilkan preview di bawah
    if uploaded_file:
        st.markdown("""<div style="margin-top: -20px;"></div>""", unsafe_allow_html=True) # Spacer adjustment
        st.markdown('<div class="css-card" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;">Gambar yang Diunggah</h4>', unsafe_allow_html=True)
        
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True)
        if st.button("🔄 Reset / Ganti Gambar", type="secondary"):
            st.session_state.uploader_key += 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. KOLOM KANAN (HASIL)
# ==========================================
with col_right:
    if uploaded_file and model:
        preds = predict_img(image, model, m_type)
        idx = np.argmax(preds[0])
        conf = np.max(preds[0]) * 100
        label = CLASS_NAMES[idx]
        data = DISEASE_KB[label]
        
        # FIX: HTML string diratakan KIRI MENTOK (tanpa spasi di awal)
        result_html = f"""
<div class="css-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
<div>
<h4 style="margin:0;">Hasil Diagnosis</h4>
<p style="font-size:12px; margin:0;">Analisis selesai</p>
</div>
<div style="background:#DCFCE7; color:#166534; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:bold;">
{data['severity']}
</div>
</div>
<div class="result-bar">{label}</div>
<div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:5px;">
<b>Kepercayaan AI</b>
<b>{conf:.1f}%</b>
</div>
<div style="background-color:#E5E7EB; height:10px; border-radius:5px; width:100%; overflow:hidden;">
<div style="background-color:#16A34A; width:{conf}%; height:100%;"></div>
</div>
<div style="margin-top:20px;">
<b>Deskripsi:</b>
<p style="font-size:14px; line-height:1.5;">{data['desc']}</p>
</div>
</div>
"""
        st.markdown(result_html, unsafe_allow_html=True)
        
        # HTML Rekomendasi (Looping items)
        rec_items = ""
        for item in data['treatments']:
            # FIX: Item loop juga diratakan kiri
            rec_items += f"""
<div style="display:flex; gap:15px; padding:15px; border-bottom:1px solid #F0FDF4;">
<div style="width:40px; height:40px; background:#F0FDF4; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:20px; border:1px solid #DCFCE7;">
{item['icon']}
</div>
<div>
<div style="font-weight:bold; font-size:14px; color:#14532D;">{item['title']}</div>
<div style="font-size:12px; color:#15803D;">{item['desc']}</div>
</div>
</div>
"""
        
        # FIX: Container Rekomendasi diratakan kiri
        rec_html = f"""
<div style="background:white; border-radius:16px; border:1px solid #DCFCE7; overflow:hidden; margin-top:20px;">
<div style="background:#059669; padding:15px 20px;">
<h4 style="margin:0; color:white !important; font-size:16px;">Rekomendasi Perawatan</h4>
<p style="margin:0; color:white !important; font-size:12px; opacity:0.9;">Langkah pencegahan & pengobatan</p>
</div>
<div>{rec_items}</div>
</div>
"""
        st.markdown(rec_html, unsafe_allow_html=True)

    elif uploaded_file and not model:
        st.warning("⚠️ Model belum dimuat! Upload di sidebar.")
    else:
        # Placeholder Kosong (Juga diratakan kiri)
        st.markdown("""
<div class="css-card" style="display:flex; align-items:center; justify-content:center; height:100%; min-height:400px; text-align:center;">
<div>
<div style="font-size:60px; opacity:0.3; margin-bottom:10px;">🍃</div>
<p style="color:#16A34A; font-weight:500;">Menunggu gambar...</p>
</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="css-card" style="padding:30px;">
    <h4 style="margin-top:0;">Cara Kerja</h4>
    <div style="display:flex; justify-content:space-between; margin-top:20px; text-align:center; gap:20px;">
        <div style="flex:1;">
            <div style="background:#DCFCE7; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-weight:bold; color:#15803D;">1</div>
            <b>Unggah Gambar</b><br><span style="font-size:13px;">Foto daun pencahayaan baik</span>
        </div>
        <div style="flex:1;">
            <div style="background:#DCFCE7; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-weight:bold; color:#15803D;">2</div>
            <b>Analisis AI</b><br><span style="font-size:13px;">Identifikasi penyakit akurat</span>
        </div>
        <div style="flex:1;">
            <div style="background:#DCFCE7; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-weight:bold; color:#15803D;">3</div>
            <b>Dapatkan Solusi</b><br><span style="font-size:13px;">Rekomendasi perawatan</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
