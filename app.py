import os
import streamlit as st

try:
    import streamlit_webrtc.shutdown
    if hasattr(streamlit_webrtc.shutdown, "SessionShutdownObserver"):
        orig_stop = streamlit_webrtc.shutdown.SessionShutdownObserver.stop
        def patched_stop(self, *args, **kwargs):
            try:
                return orig_stop(self, *args, **kwargs)
            except Exception:
                return
        streamlit_webrtc.shutdown.SessionShutdownObserver.stop = patched_stop
except Exception:
    pass

import numpy as np
import time
import pandas as pd
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="AI Proctoring", layout="wide")
st.title("🛡️ AKILLI SINAV GÜVENLİK SİSTEMİ")

st.sidebar.markdown("## 👤 GELİŞTİRİCİ BİLGİLERİ")
st.sidebar.write("**Geliştirici:** Muhammed Aydın TEKİN")
st.sidebar.write("**Proje Amacı:** Çevrimiçi sınavlarda dürüstlüğü ve sınav güvenliğini sağlamak amacıyla yapay zekâ destekli anlık yüz takibi ve ihlal tespiti gerçekleştirmek.")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Çalışma Prensibi")
st.sidebar.write("• Odada tek bir kişinin olması zorunludur.")
st.sidebar.write("• Kameradan 3 saniyeden uzun süre uzaklaşmak veya başka yöne odaklanmak ihlal sayılır.")
st.sidebar.write("• Risk skoru %100 olduğunda sistem kilitlenir ve kanıt fotoğrafı kaydeder.")

if not os.path.exists("kopya_kanitlari"):
    os.makedirs("kopya_kanitlari")

if "final_rapor_verisi" not in st.session_state:
    st.session_state.final_rapor_verisi = []
if "sinav_bitti" not in st.session_state:
    st.session_state.sinav_bitti = False
if "kilitlendi_mi" not in st.session_state:
    st.session_state.kilitlendi_mi = False
if "stream_id" not in st.session_state:
    st.session_state.stream_id = 0

class ProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.durum = "SAFE"
        self.violation_score = 0.0
        self.snapshot_saved = False
        self.init_time = time.time()
        self.history = []
        self.violation_start_time = None

    def recv(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape

        gecen_sure = time.time() - self.init_time
        kalan_hazirlik = max(0.0, 3.0 - gecen_sure)

        gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        yuzler = self.face_cascade.detectMultiScale(gri_kare, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        ihlal_tetiklendi = False

        if len(yuzler) == 1:
            (x, y, w_box, h_box) = yuzler[0]
            yuz_merkez_x = x + (w_box / 2)
            ekran_merkez_x = w / 2
            sapma = (yuz_merkez_x - ekran_merkez_x) / w

            if abs(sapma) > 0.25:
                ihlal_tetiklendi = True
        elif len(yuzler) > 1:
            ihlal_tetiklendi = True
        else:
            if kalan_hazirlik == 0:
                ihlal_tetiklendi = True

        if kalan_hazirlik > 0:
            self.durum = f"HAZIRLANIN ({kalan_hazirlik:.1f}s)"
            renk = (255, 140, 0)
        else:
            if ihlal_tetiklendi:
                if self.violation_start_time is None:
                    self.violation_start_time = time.time()
                
                if time.time() - self.violation_start_time >= 3.0:
                    self.durum = "UYARI: ŞÜPHELİ"
                    self.violation_score = min(100.0, self.violation_score + 5.0)
                    if self.violation_score >= 100.0:
                        self.durum = "KİLİTLENDİ"
            else:
                self.violation_start_time = None
                if self.violation_score > 0:
                    self.violation_score = max(0.0, self.violation_score - 4.0)
                if self.violation_score == 0.0:
                    self.durum = "SAFE"

            renk = (0, 255, 0) if "SAFE" in self.durum else ((0, 255, 255) if "UYARI" in self.durum else (0, 0, 255))

        self.history.append({"Saniye": len(self.history) * 0.1, "Risk Skoru": self.violation_score})

        cv2.rectangle(kare, (10, 10), (380, 110), (0, 0, 0), -1)
        cv2.putText(kare, f"DURUM: {self.durum}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        cv2.putText(kare, f"Risk: {self.violation_score:.1f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(kare, (20, 85), (360, 95), (50, 50, 50), -1)
        cv2.rectangle(kare, (20, 85), (20 + int(self.violation_score * 3.4), 95), renk, -1)

        if self.durum == "KİLİTLENDİ" and not self.snapshot_saved:
            cv2.imwrite(f"kopya_kanitlari/ihlal_kaniti.jpg", kare)
            self.snapshot_saved = True

        return frame.from_ndarray(kare, format="bgr24")

if not st.session_state.sinav_bitti:
    sol, sag = st.columns([2, 1])
    with sol:
        ctx = webrtc_streamer(
            key=f"proctor_stream_{st.session_state.stream_id}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ProctorProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": {"ideal": 480}, "height": {"ideal": 360}}, "audio": False},
            async_processing=True
        )
    with sag:
        st.info("Sınavı başlatmak için sol taraftaki 'Start' butonuna basın.")
        
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
        st.session_state.stream_id += 1
        if os.path.exists("kopya_kanitlari/ihlal_kaniti.jpg"):
            os.remove("kopya_kanitlari/ihlal_kaniti.jpg")
        st.rerun()
