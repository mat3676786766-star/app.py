import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Safety Monitor", layout="wide")

st.title("🌐 AI SAFETY MATRIX - REALTIME TRACKING")
st.caption("Gelişmiş Eksen Takip ve Kopya Analiz Arayüzü")

# Session State Değişkenlerini Başlatma
if "durum" not in st.session_state:
    st.session_state.durum = "SAFE"  # SAFE, KOPYA İHTİMALİ VAR, KOPYA ÇEKİYOR
if "suspicious_start" not in st.session_state:
    st.session_state.suspicious_start = None
if "grafik_verisi" not in st.session_state:
    st.session_state.grafik_verisi = []
if "prev_yaw" not in st.session_state:
    st.session_state.prev_yaw = 0.0
if "prev_pitch" not in st.session_state:
    st.session_state.prev_pitch = 0.0

# Sol ve Sağ Kolon Tasarımı
sol_kolon, sag_kolon = st.columns([2, 1])

with sol_kolon:
    st.markdown("### 📷 CANLI ANALİZ KAMERASI")
    kamera_butonu = st.toggle("Sistemi Güvenli Modda Başlat", value=False)
    kare_alani = st.empty()

with sag_kolon:
    st.markdown("### 📊 SENSÖR VERİLERİ VE GRAFİK")
    durum_alani = st.empty()
    deger_alani = st.empty()
    st.markdown("#### 📈 Gerçek Zamanlı Hareket Grafiği (Yaw & Pitch)")
    grafik_alani = st.empty()

# Kamera Döngüsü
if kamera_butonu:
    # 0 varsayılan kameradır. Açılmazsa 1 veya -1 deneyebilirsiniz.
    kamera = cv2.VideoCapture(0)
    
    # Grafik verilerini sıfırla
    st.session_state.grafik_verisi = []
    
    while kamera.isOpened():
        ret, kare = kamera.read()
        if not ret:
            st.error("Kamera akışı alınamıyor!")
            break
            
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape
        cx_ekran, cy_ekran = w // 2, h // 2
        
        # Görüntü İşleme (Kişi/Yüz tespiti için kontur analizi)
        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
        _, esik = cv2.threshold(bulanik, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # En büyük konturu bul (Kullanıcı)
        en_buyuk_kontur = None
        maks_alan = 0
        for kontur in konturlar:
            alan = cv2.contourArea(kontur)
            if alan > 10000 and alan > maks_alan:
                maks_alan = alan
                en_buyuk_kontur = kontur

        # Varsayılan Yaw ve Pitch değerleri
        yaw, pitch = st.session_state.prev_yaw, st.session_state.prev_pitch
        hareket_var = False

        if en_buyuk_kontur is not None:
            x, y, w_box, h_box = cv2.boundingRect(en_buyuk_kontur)
            cx_yuz = x + w_box // 2
            cy_yuz = y + h_box // 2
            
            # Merkez noktasına göre Yaw ve Pitch tahmini hesaplama
            yaw = float(((cx_yuz - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_yuz) / cy_ekran) * 35.0)
            
            # Anlık hareket kontrolü (Hafif kafa oynatma toleransı testi)
            if abs(yaw - st.session_state.prev_yaw) > 1.2 or abs(pitch - st.session_state.prev_pitch) > 1.2:
                hareket_var = True
                
            st.session_state.prev_yaw = yaw
            st.session_state.prev_pitch = pitch

        # --- DURUM MAKİNESİ (STATE MACHINE) MANTIĞI ---
        # Eğer sistem zaten kalıcı olarak kilitlenmediyse durumları kontrol et
        if st.session_state.durum != "KOPYA ÇEKİYOR":
            if hareket_var:
                if st.session_state.durum == "SAFE":
                    st.session_state.durum = "KOPYA İHTİMALİ VAR"
                    st.session_state.suspicious_start = time.time()
                elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    # 5 saniyeden uzun süredir hareket devam ediyor mu?
                    if time.time() - st.session_state.suspicious_start > 5.0:
                        st.session_state.durum = "KOPYA ÇEKİYOR"
            else:
                # Hareket bitti ve 5 saniye dolmadıysa tekrar SAFE durumuna dön
                if st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    st.session_state.durum = "SAFE"
                    st.session_state.suspicious_start = None

        # Renk ve Metin Atamaları
        if st.session_state.durum == "SAFE":
            renk_bgr = (0, 255, 0)       # Yeşil
            renk_hex = "#00FF00"
        elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
            renk_bgr = (0, 255, 255)     # Sarı
            renk_hex = "#FFFF00"
        else: # KOPYA ÇEKİYOR
            renk_bgr = (0, 0, 255)       # Kırmızı
            renk_hex = "#FF0000"

        # Ekrana Çizim Yapma (Kopya çekiyorsa etrafı kalıcı kırmızı kutu olur)
        if en_buyuk_kontur is not None:
            box_renk = (0, 0, 255) if st.session_state.durum == "KOPYA ÇEKİYOR" else renk_bgr
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), box_renk, 3)
            cv2.putText(kare, f"STATUS: {st.session_state.durum}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_renk, 2)

        # Sağ Panel Metin Güncellemeleri
        durum_alani.markdown(f"### DURUM: <span style='color:{renk_hex}; font-weight:bold;'>{st.session_state.durum}</span>", unsafe_allow_html=True)
        deger_alani.code(f"Anlık Yaw   : {yaw:.2f}°\nAnlık Pitch : {pitch:.2f}°")
        
        # Grafik Verisi Ekleme ve Güncelleme
        st.session_state.grafik_verisi.append({"Zaman": time.time(), "Yaw (Yatay)": yaw, "Pitch (Dikey)": pitch})
        if len(st.session_state.grafik_verisi) > 40:  # Grafikte maksimum 40 nokta tut
            st.session_state.grafik_verisi.pop(0)
            
        df = pd.DataFrame(st.session_state.grafik_verisi).set_index("Zaman")
        grafik_alani.line_chart(df)

        # Görüntüyü Streamlit paneline bas
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        
        # CPU'yu yormamak için mikro bekleme
        time.sleep(0.03)
        
    kamera.release()
