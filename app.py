import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

# Sayfa Genişlik Ayarı
st.set_page_config(page_title="AI Matrix Pro v3", layout="wide")

st.title("🌐 AI SAFETY MATRIX - HARDWARE FIXED")
st.caption("Donanım Öncelikli Eksen Kontrollü Takip Modülü")

# Session State Hafıza Yapılandırması
if "durum" not in st.session_state:
    st.session_state.durum = "SAFE"
if "suspicious_start" not in st.session_state:
    st.session_state.suspicious_start = None
if "grafik_verisi" not in st.session_state:
    st.session_state.grafik_verisi = []
if "last_yaw" not in st.session_state:
    st.session_state.last_yaw = 0.0
if "last_pitch" not in st.session_state:
    st.session_state.last_pitch = 0.0

# Sol ve Sağ Kolon Düzeni
sol_kolon, sag_kolon = st.columns([2, 1])

with sol_kolon:
    st.markdown("### 📷 CANLI ANALİZ SEKANSI")
    cam_index = st.selectbox("Kamera İndeksi:", [0, 1, 2], index=0)
    kamera_butonu = st.toggle("Sistemi Güvenli Modda Başlat", value=False)
    kare_alani = st.empty()

with sag_kolon:
    st.markdown("### 📊 GERÇEK ZAMANLI TELEMETRİ")
    durum_alani = st.empty()
    deger_alani = st.empty()
    st.markdown("---")
    st.markdown("#### 📈 Hareket Grafiği (Yaw & Pitch)")
    grafik_alani = st.empty()

# Ana Kamera Döngüsü
if kamera_butonu:
    kamera = None
    
    # --- DONANIM BAĞLANTI SİHİRBAZI (Açılma Garantili Mod) ---
    # Windows DirectShow sürücüsünü zorla (En sık yaşanan açılmama sebebi)
    kamera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    
    # Eğer DirectShow başarısız olursa standart modda açmayı dene
    if not kamera.isOpened():
        kamera = cv2.VideoCapture(cam_index)
        
    # O da yemezse MSMF (Media Foundation) dene
    if not kamera.isOpened():
        kamera = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
        
    if not kamera.isOpened():
        st.error(f"❌ KRİTİK HATA: İşletim sistemi kamera donanımına erişimi engelliyor! Kamerayı kullanan başka bir uygulama (Zoom, Chrome, Discord vb.) varsa kapatıp sayfayı yenileyin.")
    
    while kamera is not None and kamera.isOpened():
        ret, kare = kamera.read()
        if not ret or kare is None:
            # Anlık kare kaçırma durumunda hemen pes etme, döngüyü sürdür
            continue
            
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape
        cx_ekran, cy_ekran = w // 2, h // 2
        
        # Görüntü İşleme ve Kontur Analizi
        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
        _, esik = cv2.threshold(bulanik, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        en_buyuk_kontur = None
        maks_alan = 0
        for kontur in konturlar:
            alan = cv2.contourArea(kontur)
            if alan > 8000 and alan > maks_alan:
                maks_alan = alan
                en_buyuk_kontur = kontur

        # Yaw ve Pitch Hesaplamaları
        hareket_var = False
        yaw, pitch = st.session_state.last_yaw, st.session_state.last_pitch

        if en_buyuk_kontur is not None:
            x, y, w_box, h_box = cv2.boundingRect(en_buyuk_kontur)
            cx_yuz = x + w_box // 2
            cy_yuz = y + h_box // 2
            
            # Açı Tahminleri
            yaw = float(((cx_yuz - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_yuz) / cy_ekran) * 35.0)
            
            # Hassas kafa hareketi tespiti (Eşik: 1.5 derece)
            if abs(yaw - st.session_state.last_yaw) > 1.5 or abs(pitch - st.session_state.last_pitch) > 1.5:
                hareket_var = True
                
            st.session_state.last_yaw = yaw
            st.session_state.last_pitch = pitch

        # --- Durum Makinesi Mantığı ---
        if st.session_state.durum != "KOPYA ÇEKİYOR":
            if hareket_var:
                if st.session_state.durum == "SAFE":
                    st.session_state.durum = "KOPYA İHTİMALİ VAR"
                    st.session_state.suspicious_start = time.time()
                elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    # Hareket kesintisiz 5 saniyeyi geçti mi?
                    if time.time() - st.session_state.suspicious_start > 5.0:
                        st.session_state.durum = "KOPYA ÇEKİYOR"
            else:
                # Hareket 5 saniyeden önce durduysa affet ve SAFE moduna döndür
                if st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    st.session_state.durum = "SAFE"
                    st.session_state.suspicious_start = None

        # Dinamik Renk ve Metin Kodlamaları
        if st.session_state.durum == "SAFE":
            renk_bgr = (0, 255, 0)       # Yeşil
            renk_hex = "#00FF00"
        elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
            renk_bgr = (0, 255, 255)     # Sarı
            renk_hex = "#FFFF00"
        else:
            renk_bgr = (0, 0, 255)       # Kırmızı (Kalıcı)
            renk_hex = "#FF0000"

        # Görsel Çizimler
        if en_buyuk_kontur is not None:
            # Kopya çekiyorsa etrafı daima kırmızı kutu kalır
            kutu_rengi = (0, 0, 255) if st.session_state.durum == "KOPYA ÇEKİYOR" else renk_bgr
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), kutu_rengi, 3)
            cv2.putText(kare, f"{st.session_state.durum}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, kutu_rengi, 2)

        # Sağ Arayüz Paneli Çıktıları
        durum_alani.markdown(f"### DURUM: <span style='color:{renk_hex}; font-weight:bold;'>{st.session_state.durum}</span>", unsafe_allow_html=True)
        deger_alani.code(f"Anlık Yaw (Yatay) : {yaw:.2f}°\nAnlık Pitch (Dikey): {pitch:.2f}°")
        
        # Gerçek Zamanlı Grafik Güncelleme
        st.session_state.grafik_verisi.append({
            "Milisaniye": time.time() * 1000, 
            "Yaw": yaw, 
            "Pitch": pitch
        })
        
        if len(st.session_state.grafik_verisi) > 50:
            st.session_state.grafik_verisi.pop(0)
            
        df_grafik = pd.DataFrame(st.session_state.grafik_verisi).set_index("Milisaniye")
        grafik_alani.line_chart(df_grafik)

        # Görüntüyü Bas
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        
        time.sleep(0.03)
        
    if kamera is not None:
        kamera.release()
