import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class ProctorProcessor(VideoTransformerBase):
    def __init__(self):
        self.risk_orani = 0
        self.kopya_kilitlendi = False
        self.ihlal_baslangic = None
        # Grafik için veri geçmişi
        if "yaw_history" not in st.session_state: st.session_state.yaw_history = [0]*20
        if "pitch_history" not in st.session_state: st.session_state.pitch_history = [0]*20

    def transform(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        if self.kopya_kilitlendi:
            cv2.putText(kare, "KOPYA TESPIT EDILDI!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return kare

        # --- BURAYA SENİN AÇI HESAPLARINI YERLEŞTİR ---
        # Örnek: yaw ve pitch değerlerini hesapladığını varsayıyorum:
        yaw, pitch = np.random.uniform(-20, 20), np.random.uniform(-20, 20)
        
        # Grafik verisini güncelle
        st.session_state.yaw_history.append(yaw)
        st.session_state.yaw_history.pop(0)
        
        # İhtimal Mantığı: Hareket var mı?
        if abs(yaw) > 12 or abs(pitch) > 12:
            if self.ihlal_baslangic is None: self.ihlal_baslangic = time.time()
            self.risk_orani = min(100, (time.time() - self.ihlal_baslangic) * 33.3) # 3 saniyede %100'e çıkar
            durum = "KOPYA CEKME IHTIMALI VAR!"
            renk = (0, 165, 255) # Turuncu
        else:
            self.ihlal_baslangic = None
            self.risk_orani = max(0, self.risk_orani - 5)
            durum = "GUVENLI"
            renk = (0, 255, 0) # Yeşil

        # Alarm tetikleyici
        if self.risk_orani >= 100:
            self.kopya_kilitlendi = True

        cv2.putText(kare, f"DURUM: {durum}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
        return kare

st.title("🛡️ AI SAFETY MATRIX - CANLI ANALİZ")
webrtc_streamer(key="matrix", video_processor_factory=ProctorProcessor)

# Grafiklerin canlı görünmesi için
if "yaw_history" in st.session_state:
    st.line_chart(st.session_state.yaw_history)
    st.progress(min(st.session_state.get("risk", 0)/100, 1.0))
