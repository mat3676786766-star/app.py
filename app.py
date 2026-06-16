import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class ProctorProcessor(VideoTransformerBase):
    def __init__(self):
        self.risk_orani = 0.0
        self.kopya_kilitlendi = False
        self.ihlal_baslangic = None
        self.yaw_history = [0.0] * 15 # Hareket yumuşatma için
        self.pitch_history = [0.0] * 15

    def transform(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        if self.kopya_kilitlendi:
            cv2.putText(kare, "KOPYA TESPIT EDILDI!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return kare

        # --- BURAYA SENİN AÇI HESAPLARINI YERLEŞTİR ---
        # Örnek: yaw ve pitch hesaplaman
        raw_yaw = np.random.uniform(-20, 20) 
        raw_pitch = np.random.uniform(-20, 20)
        
        # Hareketleri Yumuşat (Gürültüyü engeller)
        self.yaw_history.append(raw_yaw)
        self.yaw_history.pop(0)
        self.pitch_history.append(raw_pitch)
        self.pitch_history.pop(0)
        
        yaw_smooth = sum(self.yaw_history) / len(self.yaw_history)
        pitch_smooth = sum(self.pitch_history) / len(self.pitch_history)

        # Risk Analizi (Daha kararlı eşik)
        if abs(yaw_smooth) > 12 or abs(pitch_smooth) > 12:
            if self.ihlal_baslangic is None: self.ihlal_baslangic = time.time()
            # Risk 3 saniyede %100'e ulaşsın
            self.risk_orani = min(100, (time.time() - self.ihlal_baslangic) * 33.3)
            durum = "KOPYA CEKME IHTIMALI VAR!"
            renk = (0, 165, 255)
        else:
            # GÜVENLİ durumu hemen değişmesin diye biraz yavaş düşür
            self.ihlal_baslangic = None
            self.risk_orani = max(0, self.risk_orani - 2) 
            durum = "GÜVENLİ"
            renk = (0, 255, 0)

        if self.risk_orani >= 100: self.kopya_kilitlendi = True

        # Ekrana Yazdır
        cv2.putText(kare, f"{durum} (Risk: %{int(self.risk_orani)})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)
        
        # State'e yaz (Grafik için)
        st.session_state["yaw_graph"] = yaw_smooth
        st.session_state["pitch_graph"] = pitch_smooth
        st.session_state["risk_graph"] = self.risk_orani
        
        return kare

st.title("🛡️ AI SAFETY MATRIX - CANLI ANALİZ")
webrtc_streamer(key="matrix", video_processor_factory=ProctorProcessor)

# Canlı Grafik Paneli
if "yaw_graph" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart({"Yaw": st.session_state["yaw_graph"], "Pitch": st.session_state["pitch_graph"]})
    with col2:
        st.progress(st.session_state["risk_graph"] / 100)
        st.write(f"Kopya İhtimali: %{int(st.session_state['risk_graph'])}")
