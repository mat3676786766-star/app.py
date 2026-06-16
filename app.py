import os
import sys
import subprocess
import importlib

# =====================================================================
# 🛡️ GİZLİ OPENCV ÇAKIŞMA ENGELLEYİCİ (SUNUCU MODU)
# =====================================================================
# MediaPipe'ın arkadan zorla yüklediği hatalı OpenCV'yi pasifize ediyoruz.
try:
    import cv2
except ImportError:
    # Eğer cv2 yüklenirken hata verirse, sisteme headless sürümü zorla kur ve hafızayı yenile
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "opencv-python-headless==4.9.0.80"])
    importlib.invalidate_caches()
    import cv2

import streamlit as st
import numpy as np
import time
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp

# Sayfa Ayarları
st.set_page_config(page_title="AI Proctoring", layout="wide")
st.title("🛡️ AKILLI SINAV GÜVENLİK SİSTEMİ")

# Kanıt klasörü kontrolü
if not os.path.exists("kopya_kanitlari"):
    os.makedirs("kopya_kanitlari")

if "final_rapor_verisi" not in st.session_state:
    st.session_state.final_rapor_verisi = []
if "sinav_bitti" not in st.session_state:
    st.session_state.sinav_bitti = False

# Yapay Zeka İşlem Modülü
mp_face_mesh = mp.solutions.face_mesh

class ProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.durum = "SİSTEM BAŞLATILIYOR..."
        self.violation_score = 0.0
        self.last_yaw, self.last_pitch = 0.0, 0.0
        self.snapshot_saved = False
        self.init_time = time.time()
        self.history = []

    def recv(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape

        gecen_sure = time.time() - self.init_time
        kalan_hazirlik = max(0.0, 3.0 - gecen_sure)

        rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        sonuclar = self.face_mesh.process(rgb_kare)

        ihlal_var = False
        yaw, pitch = 0.0, 0.0

        if sonuclar.multi_face_landmarks:
            yuz = sonuclar.multi_face_landmarks[0].landmark
            burun = np.array([yuz[1].x * w, yuz[1].y * h])
            sol_goz = np.array([yuz[33].x * w, yuz[33].y * h])
            sag_goz = np.array([yuz[263].x * w, yuz[263].y * h])
            alin = np.array([yuz[10].x * w, yuz[10].y * h])
            cene = np.array([yuz[152].x * w, yuz[152].y * h])

            yaw = float((np.linalg.norm(burun - sol_goz) / (np.linalg.norm(burun - sag_goz) + 1e-6) - 1.0) * 100.0)
            pitch = float((np.linalg.norm(burun - alin) / (np.linalg.norm(burun - cene) + 1e-6) - 1.2) * 100.0)
            self.last_yaw, self.last_pitch = yaw, pitch

            if abs(yaw) > 25.0 or abs(pitch) > 20.0:
                ihlal_var = True
        else:
            if kalan_hazirlik == 0:
                ihlal_var = True

        if kalan_hazirlik > 0:
            self.durum = f"HAZIRLANIN ({kalan_hazirlik:.1f}s)"
            renk = (255, 140, 0)
        else:
            if self.durum.startswith("HAZIRLANIN"): self.durum = "SAFE"
            if self.durum != "KİLİTLENDİ":
                if ihlal_var:
                    self.durum = "UYARI: ŞÜPHELİ"
                    self.violation_score = min(100.0, self.violation_score + 4.0)
                    if self.violation_score >= 100.0: self.durum = "KİLİTLENDİ"
                else:
                    self.violation_score = max(0.0, self.violation_score - 1.5)
                    if self.violation_score == 0.0: self.durum = "SAFE"

            renk = (0, 255, 0) if "SAFE" in self.durum else ((0, 255, 255) if "UYARI" in self.durum else (0, 0, 255))

        self.history.append({"Saniye": len(self.history) * 0.03, "Risk Skoru": self.violation_score})

        # Arayüz Çizimleri
        cv2.rectangle(kare, (10, 10), (380, 110), (0, 0, 0), -1)
        cv2.putText(kare, f"DURUM: {self.durum}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        cv2.putText(kare, f"Risk: {self.violation_score:.1f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(kare, (20, 85), (360, 95), (50, 50, 50), -1)
        cv2.rectangle(kare, (20, 85), (20 + int(self.violation_score * 3.4), 95), renk, -1)

        if self.durum == "KİLİTLENDİ" and not self.snapshot_saved:
            cv2.imwrite(f"kopya_kanitlari/ihlal_{int(time.time())}.jpg", kare)
            self.snapshot_saved = True

        return frame.from_ndarray(kare, format="bgr24")

# Ekran Tasarımı
if not st.session_state.sinav_bitti:
    sol, sag = st.columns([2, 1])
    with sol:
        ctx = webrtc_streamer(
            key="proctor_stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ProctorProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    with sag:
        st.info("Kamera akışını başlatmak için yukarıdaki 'Start' butonuna basın.")
        if ctx.video_processor and ctx.video_processor.durum == "KİLİTLENDİ":
            st.session_state.final_rapor_verisi = ctx.video_processor.history
            st.session_state.sinav_bitti = True
            st.rerun()
        if st.button("Sınavı Bitir", type="primary"):
            if ctx.video_processor: st.session_state.final_rapor_verisi = ctx.video_processor.history
            st.session_state.sinav_bitti = True
            st.rerun()
else:
    st.markdown("## 📊 Sınav Sonuç Raporu")
    if st.session_state.final_rapor_verisi:
        df = pd.DataFrame(st.session_state.final_rapor_verisi).set_index("Saniye")
        st.line_chart(df)
    
    if os.path.exists("kopya_kanitlari") and os.listdir("kopya_kanitlari"):
        resimler = os.listdir("kopya_kanitlari")
        st.image(os.path.join("kopya_kanitlari", resimler[-1]), caption="İhlal Kanıtı", use_container_width=True)
    else:
        st.success("Tebrikler, ihlal tespiti olmadan sınav tamamlandı.")
        
    if st.button("Yeni Sınav Başlat"):
        st.session_state.sinav_bitti = False
        st.session_state.final_rapor_verisi = []
        st.rerun()
