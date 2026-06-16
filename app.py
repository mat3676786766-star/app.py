import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Safety Matrix Pro v4", layout="wide")

st.title("🛡️ AI SAFETY MATRIX - 3 SECOND LIMIT")
st.caption("3 Saniyelik Hareket Hassasiyeti ve Doğrudan Video Akışına Gömülü HUD Paneli")

# --- WEBRTC GÖMÜLÜ İŞLEMCİ SINIFI ---
class ProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.durum = "SAFE"
        self.suspicious_start = None
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        self.prev_frame = None  # Hareket takibi için önceki kare hafızası

    def recv(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        kare = cv2.flip(kare, 1)  # Aynalama efekti
        h, w, _ = kare.shape
        cx_ekran, cy_ekran = w // 2, h // 2

        # 1. ARKA PLAN HAREKET ANALİZİ (Işıktan Bağımsız Diferansiyel)
        gray = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return frame.from_ndarray(kare, format="bgr24")

        # İki kare arasındaki farkı bul (Garantili hareket tespiti)
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        konturlar, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev_frame = gray  # Kareyi güncelle

        hareket_var = False
        yaw, pitch = self.last_yaw, self.last_pitch

        en_buyuk_kontur = None
        maks_alan = 0
        for kontur in konturlar:
            alan = cv2.contourArea(kontur)
            if alan > 3000 and alan > maks_alan:  # Hareket hassasiyet eşiği
                maks_alan = alan
                en_buyuk_kontur = kontur

        if en_buyuk_kontur is not None:
            x, y, w_box, h_box = cv2.boundingRect(en_buyuk_kontur)
            cx_hareket = x + w_box // 2
            cy_hareket = y + h_box // 2

            # Merkeze olan uzaklığa göre dinamik eksen hesaplama
            yaw = float(((cx_hareket - cx_ekran) / cx_ekran) * 45.0)
            pitch = float(((cy_ekran - cy_hareket) / cy_ekran) * 35.0)
            hareket_var = True

            self.last_yaw = yaw
            self.last_pitch = pitch
            
            # Değişim kutusunu ekrana çiz
            cv2.rectangle(kare, (x, y), (x + w_box, y + h_box), (255, 255, 255), 1)

        # 2. GELİŞMİŞ DURUM MAKİNESİ VE 3 SANİYE GERİ SAYIM MANTIĞI
        kalan_sure = 3.0  # Başlangıç limiti 3 saniyeye çekildi
        if self.durum != "KOPYA ÇEKİYOR":
            if hareket_var:
                if self.durum == "SAFE":
                    self.durum = "KOPYA İHTİMALİ VAR"
                    self.suspicious_start = time.time()
                elif self.durum == "KOPYA İHTİMALİ VAR":
                    gecen_sure = time.time() - self.suspicious_start
                    kalan_sure = max(0.0, 3.0 - gecen_sure)  # 3 saniyeden geriye sayım
                    if gecen_sure > 3.0:  # 3 saniye aşılırsa kilitlen
                        self.durum = "KOPYA ÇEKİYOR"
            else:
                # 3 saniye dolmadan hareket durursa sistemi affet ve SAFE moduna çek
                if self.durum == "KOPYA İHTİMALİ VAR":
                    self.durum = "SAFE"
                    self.suspicious_start = None

        # Renk Yönetimi
        if self.durum == "SAFE":
            renk = (0, 255, 0)      # Yeşil
        elif self.durum == "KOPYA İHTİMALİ VAR":
            renk = (0, 255, 255)    # Sarı
        else:
            renk = (0, 0, 255)      # Kırmızı

        # 3. VİDEO ÜZERİNE TELEMETRİ PANELİ (HUD ARYÜZÜ) ÇİZİMİ
        # Bilgi panelinin arka planı
        cv2.rectangle(kare, (10, 10), (380, 135), (0, 0, 0), -1)
        
        # Yazıları canlı olarak kare üzerine basıyoruz
        cv2.putText(kare, f"DURUM: {self.durum}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
        cv2.putText(kare, f"Yaw (Yatay) : {yaw:.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(kare, f"Pitch (Dikey): {pitch:.2f} deg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Sarı moddayken ekranda 3 saniyeden düşen canlı sayacı gösterir
        if self.durum == "KOPYA İHTİMALİ VAR":
            cv2.putText(kare, f"KILITLENMEYE: {kalan_sure:.1f}s", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        
        # Kırmızı moda girdiğinde tüm ekranı kalın kırmızı çerçeveye alır
        if self.durum == "KOPYA ÇEKİYOR":
            cv2.rectangle(kare, (0, 0), (w, h), (0, 0, 255), 12)
            cv2.putText(kare, "SISTEM KILITLENDI!", (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return frame.from_ndarray(kare, format="bgr24")

# --- TARAYICI TABANLI DOĞRUDAN BAĞLANTI SİHİRBAZI ---
webrtc_streamer(
    key="fixed_matrix_stream_v4",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=ProctorProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
