import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

# Sayfa Genişlik Ayarı
st.set_page_config(page_title="AI Matrix - Auto System", layout="wide")

st.title("🌐 AI SAFETY MATRIX - AUTOMATIC RUN")
st.caption("Açılışta Otomatik Başlayan Eksen, Grafik ve Durum Kilitli Analiz Sistemi")

# Hafıza Yapılandırması (İsteklerine Göre Ayarlandı)
if "durum" not in st.session_state:
    st.session_state.durum = "SAFE"  # Başlangıç durumu yeşil SAFE
if "suspicious_start" not in st.session_state:
    st.session_state.suspicious_start = None
if "grafik_verisi" not in st.session_state:
    st.session_state.grafik_verisi = []
if "last_yaw" not in st.session_state:
    st.session_state.last_yaw = 0.0
if "last_pitch" not in st.session_state:
    st.session_state.last_pitch = 0.0

# Ekran Kolon Düzeni
sol_kolon, sag_kolon = st.columns([2, 1])

with sol_kolon:
    st.markdown("### 📷 CANLI ANALİZ SEKANSI")
    # Kamera indeks seçici (Otomatik başlasa da yanlış kamera algılanırsa değiştirebilmen için kaldı)
    cam_index = st.selectbox("Kamera Donanım Kaynağı:", [0, 1, 2, -1], index=0)
    kare_alani = st.empty()

with sag_kolon:
    st.markdown("### 📊 GERÇEK ZAMANLI TELEMETRİ")
    durum_alani = st.empty()
    deger_alani = st.empty()
    st.markdown("---")
    st.markdown("#### 📈 Canlı Hareket Grafiği (Yaw & Pitch)")
    grafik_alani = st.empty()

# --- ARKA PLANDA OTOMATİK BAŞLATMA ---
# Windows DirectShow Protokolü ile donanımı zorla
kamera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

# Başarısız olursa standart modda açmayı dene
if not kamera.isOpened():
    kamera = cv2.VideoCapture(cam_index)
    
if not kamera.isOpened():
    st.error(f"⚠️ Kamera donanımına erişilemedi. Lütfen kamerayı kullanan diğer uygulamaları kapatıp yukarıdaki 'Kamera Donanım Kaynağı' listesinden farklı bir indeks deneyin.")
else:
    # Kamera açıldıysa döngü butonsuz şekilde direkt başlar
    while kamera.isOpened():
        ret, kare = kamera.read()
        if not ret or kare is None:
            continue
            
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape
        cx_ekran, cy_ekran = w // 2, h // 2
        
        # Görüntü İşleme: Kontur Analizi
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

        # Hareket Kontrol Değişkenleri
        hareket_var = False
        yaw, pitch = st.session_state.last_yaw, st.session_state.last_pitch

        if en_buyuk_kontur is not None:
            x, y, w_box, h_box = cv2.boundingRect(en_buyuk_kontur)
            cx_yuz = x + w_box // 2
            cy_yuz = y + h_box // 2
            
            # Gerçek zamanlı sürekli güncellenen Yaw ve Pitch hesaplaması
            yaw = float(((cx_yuz - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_yuz) / cy_ekran) * 35.0)
            
            # Kafayı hafif hareketlendirme ve hareketi devam ettirme hassasiyeti (Eşik: 1.2 derece)
            if abs(yaw - st.session_state.last_yaw) > 1.2 or abs(pitch - st.session_state.last_pitch) > 1.2:
                hareket_var = True
                
            st.session_state.last_yaw = yaw
            st.session_state.last_pitch = pitch

        # --- DURUM MAKİNESİ (STATE MACHINE) MANTIĞI ---
        # Bir kere "KOPYA ÇEKİYOR" moduna girdiyse bir daha asla çıkamaz (Kalıcı kilit)
        if st.session_state.durum != "KOPYA ÇEKİYOR":
            if hareket_var:
                if st.session_state.durum == "SAFE":
                    st.session_state.durum = "KOPYA İHTİMALİ VAR"
                    st.session_state.suspicious_start = time.time()
                elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    # Hareket aralıksız 5 saniyeyi geçtiyse "KOPYA ÇEKİYOR" moduna sabitle
                    if time.time() - st.session_state.suspicious_start > 5.0:
                        st.session_state.durum = "KOPYA ÇEKİYOR"
            else:
                # Kopya ihtimalindeyken 5 saniyeyi geçmediyse tekrar SAFE yazabilir
                if st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    st.session_state.durum = "SAFE"
                    st.session_state.suspicious_start = None

        # Duruma Göre Renk Atamaları (Yeşil, Sarı, Kırmızı)
        if st.session_state.durum == "SAFE":
            renk_bgr = (0, 255, 0)       # Yeşil
            renk_hex = "#00FF00"
        elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
            renk_bgr = (0, 255, 255)     # Sarı
            renk_hex = "#FFFF00"
        else:
            renk_bgr = (0, 0, 255)       # Kırmızı
            renk_hex = "#FF0000"

        # Görsel Çizim Kuralları
        if en_buyuk_kontur is not None:
            # ŞART: Bir kez kopya çekiyora girdikten sonra etrafı HER ZAMAN kırmızı kalacak
            kutu_rengi = (0, 0, 255) if st.session_state.durum == "KOPYA ÇEKİYOR" else renk_bgr
            
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), kutu_rengi, 3)
            cv2.putText(kare, f"{st.session_state.durum}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, kutu_rengi, 2)

        # Sağ Panel Metin Alanlarının Sürekli Güncellenmesi
        durum_alani.markdown(f"### DURUM: <span style='color:{renk_hex}; font-weight:bold;'>{st.session_state.durum}</span>", unsafe_allow_html=True)
        deger_alani.code(f"Sürekli Güncel Yaw   : {yaw:.2f}°\nSürekli Güncel Pitch : {pitch:.2f}°")
        
        # Gerçek Zamanlı Grafik Güncelleme Verisi
        st.session_state.grafik_verisi.append({
            "Zaman Saniyesi": time.time(), 
            "Yaw (Yatay Eksendeki Hareket)": yaw, 
            "Pitch (Dikey Eksendeki Hareket)": pitch
        })
        
        # Grafiğin şişmemesi için ekranda son 40 veri noktasını tut
        if len(st.session_state.grafik_verisi) > 40:
            st.session_state.grafik_verisi.pop(0)
            
        df_grafik = pd.DataFrame(st.session_state.grafik_verisi).set_index("Zaman Saniyesi")
        grafik_alani.line_chart(df_grafik)

        # Görüntüyü Arayüze Bas
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        
        # CPU rahatlatma gecikmesi
        time.sleep(0.02)
        
    kamera.release()
