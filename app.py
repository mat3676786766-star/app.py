import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

# Sayfa Genişlik Ayarı
st.set_page_config(page_title="AI Matrix Master", layout="wide")

st.title("🌐 AI SAFETY MATRIX - EXACT LOGIC")
st.caption("Eksen Kontrollü, Sürekli Grafik ve Durum Kilitli Analiz Modülü")

# Session State Yapılandırması (İsteklerine Göre Ayarlandı)
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
    cam_index = st.selectbox("Kamera Donanım İndeksi (0 en yaygın olanıdır):", [0, 1, 2, -1], index=0)
    kamera_butonu = st.toggle("Sistemi Güvenli Modda Başlat", value=False)
    kare_alani = st.empty()

with sag_kolon:
    st.markdown("### 📊 GERÇEK ZAMANLI TELEMETRİ")
    durum_alani = st.empty()
    deger_alani = st.empty()
    st.markdown("---")
    st.markdown("#### 📈 Canlı Hareket Grafiği (Yaw & Pitch)")
    grafik_alani = st.empty()

# Ana Kamera Çalışma Döngüsü
if kamera_butonu:
    # Standart başlatma
    kamera = cv2.VideoCapture(cam_index)
    
    # Windows DirectShow Protokolü ile zorlama (Kilit kırma denemesi)
    if not kamera.isOpened():
        kamera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        
    if not kamera.isOpened():
        st.error(f"❌ SİSTEM ENGELLİ: Bilgisayarınız Python'ın kameraya erişmesine izin vermiyor. Lütfen Windows/Mac gizlilik ayarlarından kamera izinlerini kontrol edin ve kamerayı kullanan tüm arka plan uygulamalarını kapatın.")
    
    while kamera.isOpened():
        ret, kare = kamera.read()
        if not ret or kare is None:
            continue
            
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape
        cx_ekran, cy_ekran = w // 2, h // 2
        
        # Görüntü İşleme: Yüz Alanı Tespiti için Kontur Analizi
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
            
            # Gerçek zamanlı Yaw ve Pitch güncellemeleri (Ekran merkezine göre oranlama)
            yaw = float(((cx_yuz - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_yuz) / cy_ekran) * 35.0)
            
            # Kafayı hafif hareketlendirme ve hareketi devam ettirme hassasiyeti (Eşik: 1.2 derece)
            if abs(yaw - st.session_state.last_yaw) > 1.2 or abs(pitch - st.session_state.last_pitch) > 1.2:
                hareket_var = True
                
            st.session_state.last_yaw = yaw
            st.session_state.last_pitch = pitch

        # --- İSTEDİĞİN DURUM MAKİNESİ (STATE MACHINE) MANTIĞI ---
        # Bir kere "KOPYA ÇEKİYOR" moduna girdiyse bir daha asla çıkamaz (Kalıcı kilit)
        if st.session_state.durum != "KOPYA ÇEKİYOR":
            if hareket_var:
                if st.session_state.durum == "SAFE":
                    st.session_state.durum = "KOPYA İHTİMALİ VAR"
                    st.session_state.suspicious_start = time.time()
                elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    # Hareket aralıksız 5 saniyeyi geçtiyse "KOPYA ÇEKİYOR" moduna geçir
                    if time.time() - st.session_state.suspicious_start > 5.0:
                        st.session_state.durum = "KOPYA ÇEKİYOR"
            else:
                # Kopya ihtimalindeyken 5 saniyeyi geçmediyse ve hareket durduysa tekrar SAFE yazabilir
                if st.session_state.durum == "KOPYA İHTİMALİ VAR":
                    st.session_state.durum = "SAFE"
                    st.session_state.suspicious_start = None

        # Renk Kombinasyonları (İsteklerine göre tam uyumlu)
        if st.session_state.durum == "SAFE":
            renk_bgr = (0, 255, 0)       # Yeşil (BGR)
            renk_hex = "#00FF00"         # Yeşil (Hex)
        elif st.session_state.durum == "KOPYA İHTİMALİ VAR":
            renk_bgr = (0, 255, 255)     # Sarı (BGR)
            renk_hex = "#FFFF00"         # Sarı (Hex)
        else: # KOPYA ÇEKİYOR
            renk_bgr = (0, 0, 255)       # Kırmızı (BGR)
            renk_hex = "#FF0000"         # Kırmızı (Hex)

        # Görsel Çizim Kuralları
        if en_buyuk_kontur is not None:
            # İSTİSNA: Bir kez kopya çekiyora girdikten sonra etrafı HER ZAMAN kırmızı olacak
            kutu_rengi = (0, 0, 255) if st.session_state.durum == "KOPYA ÇEKİYOR" else renk_bgr
            
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), kutu_rengi, 3)
            cv2.putText(kare, f"{st.session_state.durum}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, kutu_rengi, 2)

        # Sağ Panel Metin Alanlarının Dinamik Güncellenmesi
        durum_alani.markdown(f"### DURUM: <span style='color:{renk_hex}; font-weight:bold;'>{st.session_state.durum}</span>", unsafe_allow_html=True)
        deger_alani.code(f"Sürekli Güncel Yaw   : {yaw:.2f}°\nSürekli Güncel Pitch : {pitch:.2f}°")
        
        # Gerçek Zamanlı Grafik Güncelleme Mantığı (Sürekli Akar)
        st.session_state.grafik_verisi.append({
            "Zaman Damgası": time.time() * 1000, 
            "Yaw (Yatay Eksen)": yaw, 
            "Pitch (Dikey Eksen)": pitch
        })
        
        # Grafiğin şişip tarayıcıyı dondurmaması için son 40 kareyi göster
        if len(st.session_state.grafik_verisi) > 40:
            st.session_state.grafik_verisi.pop(0)
            
        df_grafik = pd.DataFrame(st.session_state.grafik_verisi).set_index("Zaman Damgası")
        grafik_alani.line_chart(df_grafik)

        # Görüntüyü Web Arayüzüne Yansıt
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        
        # Kararlı çalışma gecikmesi
        time.sleep(0.02)
        
    kamera.release()
