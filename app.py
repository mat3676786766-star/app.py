import streamlit as st
import cv2
import numpy as np
import time
import random
from datetime import datetime

# Sayfa Genişlik Ayarları
st.set_page_config(page_title="AI Safety Matrix v5.2", layout="wide")

FILTRE_BOYUTU = 5          
SINAV_SURESI_DAKIKA = 10  
TOLERANS_SURESI = 3.0

# Log Fonksiyonu
if "loglar" not in st.session_state:
    st.session_state.loglar = ["[SYSTEM ENTERING SMOOTH TOLERANCE MATRIX (v5.2)]"]

def web_log_kaydet(mesaj):
    zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.loglar.append(f"[{zaman_damgasi}] {mesaj}")

# Oturum Değişkenleri
if "sistem_baslatildi" not in st.session_state:
    st.session_state.update({
        "hareket_baslangic": None, "ekrandan_ayrilma": None, "coklu_kisi_baslangic": None,
        "durum_mesaji": "SISTEM HAZIRLANIYOR...", "renk": (0, 255, 0), "kopya_kilitlendi": False,
        "yaw_gecmisi": [], "pitch_gecmisi": [], "kalibrasyon_bitti": False, "kalibrasyon_baslangic": time.time(),
        "cal_yaw_offset": 0, "cal_pitch_offset": 0, "cal_yaw_list": [], "cal_pitch_list": [],
        "son_kontrol_zamani": time.time(), "kontrol_aktif": False, "kontrol_bekleme_suresi": random.randint(15, 25),
        "onay_suresi_baslangic": None, "sinav_baslangic_zamani": time.time(), "son_ortam_parlakligi": None,
        "isik_ihlal_baslangici": None, "sabotaj_baslangici": None, "yasakli_bolge_baslangici": None,
        "arka_plan_ihlal_baslangici": None, "risk_orani": 0.0,
        "isik_durumu_metni": "STABLE LIGHT", "images_arka_plan": None, "son_goz_kirpma_zamani": time.time(),
        "eski_gri_kare": None, "mikro_titresim_gecmisi": []
    })

st.title("🌐 CYBER DEFENSE - ULTIMATE AI SAFETY MATRIX (v5.2)")
st.caption("Yapay Zekâ Destekli Gelişmiş Sınav Güvenliği ve Prosedür Matrisi")

sol_kolon, sag_kolon = st.columns([3, 1])

with sol_kolon:
    st.markdown("### 📷 CANLI SENSÖR VE ANALİZ KAMERASI")
    kamera_butonu = st.toggle("Sistemi Güvenli Modda Başlat", value=False)
    kare_alani = st.empty()
    canlilik_butonu_alani = st.empty()

with sag_kolon:
    st.markdown("### 🖥️ PROCTOR MATRIX PANEL")
    kalan_sure_alani = st.empty()
    kitle_sayisi_alani = st.empty()
    sensor_durumu_alani = st.empty()
    risk_bar_alani = st.empty()
    log_alani = st.empty()

# --- GÜNCEL KONTROL MANTIĞI ---
if kamera_butonu and not st.session_state.kopya_kilitlendi:
    kamera = cv2.VideoCapture(0)
    ret, kare = kamera.read()
    
    if ret:
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape
        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        gri_bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
        
        # [BURAYA ESKİ DÖNGÜ İÇİNDEKİ ANALİZ KODLARINIZ GELECEK]
        # (Mikro titreşim, sabotaj, ışık, kontur, kalibrasyon mantığı aynı şekilde duracak)
        
        # Görüntü İşleme ve Arayüz Güncelleme
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        
        # Güncelleme
        gecen_sure = time.time() - st.session_state.sinav_baslangic_zamani
        toplam_saniye = max(0, (SINAV_SURESI_DAKIKA * 60) - gecen_sure)
        kalan_sure_alani.metric("⏱️ Kalan Süre", f"{int(toplam_saniye // 60):02d}:{int(toplam_saniye % 60):02d}")
        risk_bar_alani.progress(int(st.session_state.risk_orani) / 100.0)
        log_alani.text_area("Canlı Akış", value="\n".join(st.session_state.loglar[-8:]), height=200)

        kamera.release()
        time.sleep(0.05)
        st.rerun() # DÖNGÜ YERİNE SAYFAYI YENİLİYORUZ
    else:
        st.warning("Kameraya erişilemiyor!")
        kamera.release()
