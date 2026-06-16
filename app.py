import os
import streamlit as st
import numpy as np
import time
import pandas as pd
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Proctoring", layout="wide")
st.title("🛡️ AKILLI SINAV GÜVENLİK SİSTEMİ")

# Klasör kontrolü
if not os.path.exists("kopya_kanitlari"):
    os.makedirs("kopya_kanitlari")

# Session State Tanımlamaları
if "final_rapor_verisi" not in st.session_state:
    st.session_state.final_rapor_verisi = []
if "sinav_bitti" not in st.session_state:
    st.session_state.sinav_bitti = False
if "kilitlendi_mi" not in st.session_state:
    st.session_state.kilitlendi_mi = False

mp_face_mesh = mp.solutions.face_mesh

class ProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.durum = "SAFE"
        self.violation_score = 0.0
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

        if sonuclar.multi_face_landmarks:
            yuz = sonuclar.multi_face_landmarks[0].landmark
            burun = np.array([yuz[1].x * w, yuz[1].y * h])
            sol_goz = np.array([yuz[33].x * w, yuz[33].y * h])
            sag_goz = np.array([yuz[263].x * w, yuz[263].y * h])
            alin = np.array([yuz[10].x * w, yuz[10].y * h])
            cene = np.array([yuz[152].x * w, yuz[152].y * h])

            yaw = float((np.linalg.norm(burun - sol_goz) / (np.linalg.norm(burun - sag_goz) + 1e-6) - 1.0) * 100.0)
            pitch = float((np.linalg.norm(burun - alin) / (np.linalg.norm(burun - cene) + 1e-6) - 1.2) * 100.0)

            if abs(yaw) > 25.0 or abs(pitch) > 20.0:
                ihlal_var = True
        else:
            if kalan_hazirlik == 0:
                ihlal_var = True

        if kalan_hazirlik > 0:
            self.durum = f"HAZIRLANIN ({kalan_hazirlik:.1f}s)"
            renk = (255, 140, 0)
        else:
            if ihlal_var:
                self.durum = "UYARI: ŞÜPHELİ"
                self.violation_score = min(100.0, self.violation_score + 5.0) # Mobilde hızlı tepki için biraz artırıldı
                if self.violation_score >= 100.0: 
                    self.durum = "KİLİTLENDİ"
            else:
                if self.violation_score > 0:
                    self.violation_score = max(0.0, self.violation_score - 2.0)
                if self.violation_score == 0.0: 
                    self.durum = "SAFE"

            renk = (0, 255, 0) if "SAFE" in self.durum else ((0, 255, 255) if "UYARI" in self.durum else (0, 0, 255))

        self.history.append({"Saniye": len(self.history) * 0.1, "Risk Skoru": self.violation_score})

        # Ekran Çizimleri
        cv2.rectangle(kare, (10, 10), (380, 110), (0, 0, 0), -1)
        cv2.putText(kare, f"DURUM: {self.durum}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        cv2.putText(kare, f"Risk: {self.violation_score:.1f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(kare, (20, 85), (360, 95), (50, 50, 50), -1)
        cv2.rectangle(kare, (20, 85), (20 + int(self.violation_score * 3.4), 95), renk, -1)

        if self.durum == "KİLİTLENDİ" and not self.snapshot_saved:
            cv2.imwrite(f"kopya_kanitlari/ihlal_kaniti.jpg", kare)
            self.snapshot_saved = True

        return frame.from_ndarray(kare, format="bgr24")

# UI Yönetimi
if not st.session_state.sinav_bitti:
    sol, sag = st.columns([2, 1])
    with sol:
        ctx = webrtc_streamer(
            key="proctor_stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ProctorProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"], "urls": ["stun:stun1.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False}, # Telefonlar için ideal çözünürlük
            async_processing=True
        )
    with sag:
        st.info("Sınavı başlatmak için sol taraftaki 'Start' butonuna basın.")
        
        # Sürekli rerun yapmak yerine kullanıcıya durum kilitlendiğinde aksiyon aldırıyoruz
        if ctx.video_processor:
            if ctx.video_processor.durum == "KİLİTLENDİ" and not st.session_state.kilitlendi_mi:
                st.session_state.final_rapor_verisi = ctx.video_processor.history
                st.session_state.kilitlendi_mi = True
                st.error("⚠️ SİSTEM KİLİTLENDİ! İhlal tespit edildi.")
                if st.button("Sonuç Raporunu Göster"):
                    st.session_state.sinav_bitti = True
                    st.rerun()
        
        if st.button("Sınavı Manuel Bitir", type="primary"):
            if ctx.video_processor: 
                st.session_state.final_rapor_verisi = ctx.video_processor.history
            st.session_state.sinav_bitti = True
            st.rerun()
else:
    st.markdown("## 📊 Sınav Sonuç Raporu")
    if st.session_state.final_rapor_verisi:
        df = pd.DataFrame(st.session_state.final_rapor_verisi)
        if not df.empty and "Saniye" in df.columns:
            df = df.set_index("Saniye")
            st.line_chart(df)
    
    if os.path.exists("kopya_kanitlari/ihlal_kaniti.jpg"):
        st.image("kopya_kanitlari/ihlal_kaniti.jpg", caption="Sistem Tarafından Yakalanan İhlal Kanıtı", use_container_width=True)
    else:
        st.success("Tebrikler, herhangi bir güvenlik ihlali olmadan sınav tamamlandı.")
        
    if st.button("Yeni Sınav Başlat"):
        st.session_state.sinav_bitti = False
        st.session_state.kilitlendi_mi = False
        st.session_state.final_rapor_verisi = []
        if os.path.exists("kopya_kanitlari/ihlal_kaniti.jpg"):
            os.remove("kopya_kanitlari/ihlal_kaniti.jpg")
        st.rerun()
