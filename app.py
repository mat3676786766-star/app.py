import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- GÜVENLİK ANALİZ SINIFI ---
class ProctorProcessor(VideoTransformerBase):
    def __init__(self):
        self.risk_orani = 0.0
        self.kopya_kilitlendi = False
        self.ihlal_baslangic = None
        self.yaw, self.pitch = 0.0, 0.0

    def transform(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        if self.kopya_kilitlendi:
            cv2.putText(kare, "KOPYA TESPIT EDILDI!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return kare

        # Açısal Hesaplama (Basit Örnek - Senin 300 satırlık formülleri buraya ekle)
        h, w = kare.shape[:2]
        self.yaw = (np.random.rand() - 0.5) * 30 # Buraya kendi yaw formülünü yaz
        self.pitch = (np.random.rand() - 0.5) * 30 # Buraya kendi pitch formülünü yaz

        # Risk Mantığı: Hareket var mı?
        if abs(self.yaw) > 10 or abs(self.pitch) > 10:
            self.risk_orani = min(100, self.risk_orani + 2)
        else:
            self.risk_orani = max(0, self.risk_orani - 1)

        # 3 Saniye Kuralı
        if self.risk_orani > 70:
            if self.ihlal_baslangic is None: self.ihlal_baslangic = time.time()
            if time.time() - self.ihlal_baslangic > 3.0: self.kopya_kilitlendi = True
            durum = "KOPYA CEKME IHTIMALI VAR!"
            renk = (0, 165, 255)
        else:
            self.ihlal_baslangic = None
            durum = "GÜVENLİ"
            renk = (0, 255, 0)

        cv2.putText(kare, f"DURUM: {durum}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
        
        # Verileri global state'e yaz (Grafik için)
        st.session_state["yaw"] = self.yaw
        st.session_state["pitch"] = self.pitch
        st.session_state["risk"] = self.risk_orani
        
        return kare

# --- ARAYÜZ ---
st.title("🛡️ AI SAFETY MATRIX v5.2")

# Kamera akışını başlat
webrtc_streamer(key="matrix", video_processor_factory=ProctorProcessor)

# --- CANLI GRAFİKLER ---
if "yaw" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Hareket Grafiği (Yaw/Pitch)")
        # Anlık verileri biriktirerek çizdir
        data = np.array([st.session_state.get("yaw", 0), st.session_state.get("pitch", 0)])
        st.line_chart(data)
        
    with col2:
        st.subheader("🔥 Risk Analizi")
        st.progress(st.session_state.get("risk", 0) / 100)
        st.write(f"Kopya İhtimali: %{int(st.session_state.get('risk', 0))}")

st.info("Hareket algılandığında sistem tetiklenir, 3 saniye sürerse kilitlenir.")
