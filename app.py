import streamlit as st
import cv2
import numpy as np
import time
import random
from datetime import datetime

# Sayfa Genişlik Ayarları
st.set_page_config(page_title="AI Safety Matrix v5.2", layout="wide")

# ... (Senin sabitlerin ve log fonksiyonun burada aynen kalsın) ...
FILTRE_BOYUTU = 5          
SINAV_SURESI_DAKIKA = 10  
TOLERANS_SURESI = 3.0

if "loglar" not in st.session_state:
    st.session_state.loglar = ["[SYSTEM READY]"]

def web_log_kaydet(mesaj):
    zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.loglar.append(f"[{zaman_damgasi}] {mesaj}")

# ... (Session state ayarların aynen kalsın) ...
if "sistem_baslatildi" not in st.session_state:
    st.session_state.update({
        "hareket_baslangic": None, "ekrandan_ayrilma": None, "coklu_kisi_baslangic": None,
        "durum_mesaji": "SISTEM HAZIR...", "renk": (0, 255, 0), "kopya_kilitlendi": False,
        "yaw_gecmisi": [], "pitch_gecmisi": [], "kalibrasyon_bitti": False, "kalibrasyon_baslangic": time.time(),
        "cal_yaw_offset": 0, "cal_pitch_offset": 0, "cal_yaw_list": [], "cal_pitch_list": [],
        "son_kontrol_zamani": time.time(), "kontrol_aktif": False, "risk_orani": 0.0,
        "eski_gri_kare": None, "mikro_titresim_gecmisi": [], "sinav_baslangic_zamani": time.time()
    })

st.title("🌐 CYBER DEFENSE - AI SAFETY MATRIX")

# 1. KAMERA GİRİŞİ (St.camera_input en doğrusudur)
kamera_verisi = st.camera_input("Sistemi Başlatmak İçin Kameraya İzin Verin")

if kamera_verisi is not None and not st.session_state.kopya_kilitlendi:
    # Görüntüyü OpenCV formatına çevir
    bytes_data = kamera_verisi.getvalue()
    kare = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # --- SENİN 300 SATIRLIK MANTIĞIN BURAYA ---
    # Sadece 'kamera.read()' yerine yukarıdaki 'kare' değişkenini kullan.
    kare = cv2.flip(kare, 1)
    
    # ... (Buraya o 300 satırlık analiz kodlarını yerleştir) ...
    # 'kamera.release()' ve 'while' döngüsüne ihtiyacın kalmadı!
    
    # Görüntüyü ekrana bas
    st.image(cv2.cvtColor(kare, cv2.COLOR_BGR2RGB), channels="RGB")
    
    # Sayfayı sürekli yenile
    time.sleep(0.1)
    st.rerun()

elif st.session_state.kopya_kilitlendi:
    st.error("❌ MATRİS KİLİTLENDİ!")
