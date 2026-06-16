import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

# Sayfa Genişlik Ayarı
st.set_page_config(page_title="AI Matrix Pro", layout="wide")

st.title("🌐 AI SAFETY MATRIX - TRACKING SYSTEM")
st.caption("Eksen Kontrollü ve Durum Kilitli Canlı Analiz Modülü")

# Session State Hafıza Yapılandırması (Kalıcı Durumlar İçin)
if "durum" not in st.session_state:
    st.session_state.durum = "SAFE"  # Başlangıçta SAFE (Yeşil)
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
    
    # Donanımsal kilitlenmeleri önlemek için kamera indeksi seçici
    cam_index = st.selectbox("Kamera Kaynağı (Açılmazsa Değiştirin):", [0, 1, 2], index=0)
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
    kamera = cv2.VideoCapture(cam_index)
    
    # Donanım kontrolü
    if not kamera.isOpened():
        st.error(f"Kamera (Index: {cam_index}) başlatılamadı! Lütfen diğer indeksi seçin veya izinleri kontrol edin.")
    
    while kamera.isOpened():
        ret, kare = kamera.read()
        if not ret:
            st.warning("Kamera akışından kare okunamıyor...")
            break
            
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
            
            # Merkeze olan uzaklığa göre anlık açı tahmini
            yaw = float(((cx_yuz - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_yuz) / cy_ekran) * 35.0)
            
            # Hassas kafa hareketi tespiti (Eşik değeri: 1.5 derece)
            if abs(yaw - st.session_state.last_yaw) > 1.5 or abs(pitch - st.session_state.last_pitch) > 1.5:
                hareket_var = True
                
            st.session_state.last_yaw = yaw
            st.session_state.last_pitch = pitch

        # --- Gelişmiş Durum Makinesi (State Machine) Mantığı ---
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
                # Hareket 5 saniyeden önce durduysa tekrar SAFE moduna dönme izni
                if st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    st.session_state.durum = "SAFE"
                    st.session_state.suspicious_start = None

        # Renk Kodları ve Grafik Metin Ayarları
        if st.session_state.durum == "SAFE":
            renk_bgr = (0, 255, 0)       # Yeşil
            renk_hex = "#00FF00"
        elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
            renk_bgr = (0, 255, 255)     # Sarı
            renk_hex = "#FFFF00"
        else:
            renk_bgr = (0, 0, 255)       # Kırmızı (Kalıcı Kilit)
            renk_hex = "#FF0000"

        # Ekrana Kutuları Çizme
        if en_buyuk_kontur is not None:
            # Eğer bir kez KOPYA ÇEKİYOR olduysa, etrafı tamamen kalıcı kırmızı kutu olur
            kutu_rengi = (0, 0, 255) if st.session_state.durum == "KOPYA ÇEKİYOR" else renk_bgr
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), kutu_rengi, 3)
            cv2.putText(kare, f"{st.session_state.durum}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, kutu_rengi, 2)

        # Sağ Arayüz Paneli Güncellemeleri
        durum_alani.markdown(f"### DURUM: <span style='color:{renk_hex}; font-weight:bold;'>{st.session_state.durum}</span>", unsafe_allow_html=True)
        deger_alani.code(f"Anlık Yaw (Yatay) : {yaw:.2f}°\nAnlık Pitch (Dikey): {pitch:.2f}°")
        
        # Gerçek Zamanlı Grafik Matrisi Veri Girişi
        st.session_state.grafik_verisi.append({
            "Milisaniye": time.time() * 1000, 
            "Yaw (Yatay Eksendeki Hareket)": yaw, 
            "Pitch (Dikey Eksendeki Hareket)": pitch
        })
        
        # Grafik akışının şişmemesi için son 50 kareyi hafızada tut
        if len(st.session_state.grafik_verisi) > 50:
            st.session_state.grafik_verisi.pop(0)
            
        df_grafik = pd.DataFrame(st.session_state.grafik_verisi).set_index("Milisaniye")
        grafik_alani.line_chart(df_grafik)

        # Görüntüyü Web Arayüzüne Aktarma
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        
        # İşlemciyi korumak için kararlı çalışma gecikmesi
        time.sleep(0.02)
        
    kamera.release()
