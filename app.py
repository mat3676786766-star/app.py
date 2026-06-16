import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Safety Matrix - Pure Auto", layout="wide")

st.title("🌐 AI SAFETY MATRIX - FULLY AUTOMATED")
st.caption("Sıfır Ayar, Doğrudan Tarayıcı Tabanlı Canlı Takip ve Durum Kilit Sistemi")

# Grafik verileri için session state hafızası
if "grafik_gecmisi" not in st.session_state:
    st.session_state.grafik_gecmisi = []

# --- WEBRTC VİDEO İŞLEME SINIFI ---
class ProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.durum = "SAFE"  # SAFE, KOPYA İHTİMALİ VAR, KOPYA ÇEKİYOR
        self.suspicious_start = None
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        
        # Ana thread için anlık değer aktarıcıları
        self.current_yaw = 0.0
        self.current_pitch = 0.0

    def recv(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        kare = cv2.flip(kare, 1) # Aynalama
        h, w, _ = kare.shape
        cx_ekran, cy_ekran = w // 2, h // 2

        # 1. Görüntü İşleme: Kontur Analizi ile Kafa/Yüz Alanı Bulma
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

        # 2. Yaw / Pitch ve Hareket Kontrolü
        hareket_var = False
        yaw, pitch = self.last_yaw, self.last_pitch

        if en_buyuk_kontur is not None:
            x, y, w_box, h_box = cv2.boundingRect(en_buyuk_kontur)
            cx_yuz = x + w_box // 2
            cy_yuz = y + h_box // 2
            
            # Merkeze göre sürekli güncellenen eksen değerleri
            yaw = float(((cx_yuz - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_yuz) / cy_ekran) * 35.0)
            
            # Kafayı hafif hareketlendirme ve hareketi devam ettirme hassasiyeti (Eşik: 1.2 derece)
            if abs(yaw - self.last_yaw) > 1.2 or abs(pitch - self.last_pitch) > 1.2:
                hareket_var = True
                
            self.last_yaw = yaw
            self.last_pitch = pitch

        # Dış dünyaya verileri aktar
        self.current_yaw = yaw
        self.current_pitch = pitch

        # 3. İstenen Durum Makinesi Mantığı
        # Bir kez "KOPYA ÇEKİYOR" moduna girdiyse bir daha asla çıkamaz (Kalıcı Kilit)
        if self.durum != "KOPYA ÇEKİYOR":
            if hareket_var:
                if self.durum == "SAFE":
                    self.durum = "KOPYA İHTİMALİ VAR"
                    self.suspicious_start = time.time()
                elif self.durum == "KOPYA İHTİMALİ VAR":
                    # Hareket kesintisiz 5 saniyeyi geçerse durulamaz kırmızı moda girer
                    if time.time() - self.suspicious_start > 5.0:
                        self.durum = "KOPYA ÇEKİYOR"
            else:
                # 5 saniyeyi geçmediyse ve hareket durduysa tekrar SAFE yazabilir
                if self.durum == "KOPYA İHTİMALİ VAR":
                    self.durum = "SAFE"
                    self.suspicious_start = None

        # Renk Atamaları
        if self.durum == "SAFE":
            renk_bgr = (0, 255, 0)       # Yeşil
        elif self.durum == "KOPYA İHTİMALİ VAR":
            renk_bgr = (0, 255, 255)     # Sarı
        else:
            renk_bgr = (0, 0, 255)       # Kırmızı

        # 4. Görsel Sınır Çizgileri
        if en_buyuk_kontur is not None:
            # ŞART: Bir kere kopya çekiyora girdikten sonra etrafı HER ZAMAN kırmızı kalacak
            kutu_rengi = (0, 0, 255) if self.durum == "KOPYA ÇEKİYOR" else renk_bgr
            
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), kutu_rengi, 3)
            cv2.putText(kare, f"{self.durum}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, kutu_rengi, 2)

        return frame.from_ndarray(kare, format="bgr24")

# --- SAF ARAYÜZ TASARIMI ---
sol_kolon, sag_kolon = st.columns([2, 1])

with sol_kolon:
    st.markdown("### 📷 CANLI VİDEO AKIŞI")
    # Hiçbir buton veya kaynak seçimi yok, doğrudan tarayıcı kamerasını çağırır
    ctx = webrtc_streamer(
        key="pure_matrix_stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ProctorProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with sag_kolon:
    st.markdown("### 📊 GERÇEK ZAMANLI TELEMETRİ")
    durum_alani = st.empty()
    deger_alani = st.empty()
    st.markdown("---")
    st.markdown("#### 📈 Gerçek Zamanlı Hareket Grafiği (Yaw & Pitch)")
    grafik_alani = st.empty()

# --- ARKA PLAN VERİ OKUMA VE GRAFİK DÖNGÜSÜ ---
if ctx.video_processor:
    while ctx.state.playing:
        processor = ctx.video_processor
        if processor is not None:
            mevcut_durum = processor.durum
            y_val = processor.current_yaw
            p_val = processor.current_pitch

            # HTML yazı renk kodları
            if mevcut_durum == "SAFE":
                renk_hex = "#00FF00" # Yeşil
            elif mevcut_durum == "KOPYA İHTİMALİ VAR":
                renk_hex = "#FFFF00" # Sarı
            else:
                renk_hex = "#FF0000" # Kırmızı

            # Sağ paneli anlık besle
            durum_alani.markdown(f"### DURUM: <span style='color:{renk_hex}; font-weight:bold;'>{mevcut_durum}</span>", unsafe_allow_html=True)
            deger_alani.code(f"Anlık Yaw (Yatay)  : {y_val:.2f}°\nAnlık Pitch (Dikey) : {p_val:.2f}°")

            # Grafiği güncelle
            st.session_state.grafik_gecmisi.append({
                "Saniye": time.time(),
                "Yaw (Yatay)": y_val,
                "Pitch (Dikey)": p_val
            })

            # Grafiğin aşırı birikmesini önlemek için son 40 veri noktasında kes
            if len(st.session_state.grafik_gecmisi) > 40:
                st.session_state.grafik_gecmisi.pop(0)

            df_grafik = pd.DataFrame(st.session_state.grafik_gecmisi).set_index("Saniye")
            grafik_alani.line_chart(df_grafik)

        time.sleep(0.1)
