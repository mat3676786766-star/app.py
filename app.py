import os
import sys
import subprocess

# =====================================================================
# 🔥 SUNUCU KORUMA VE KENDİ KENDİNİ TAMİR ETME SİSTEMİ (NÜKLEER ÇÖZÜM) 🔥
# =====================================================================
# Streamlit Cloud'un grafik kütüphanesi hatası vermesini doğrudan kod içinden engelliyoruz.
try:
    import cv2
except (ImportError, Exception):
    # Eğer standart OpenCV çökme yaratırsa, sunucudaki tüm OpenCV kalıntılarını kazı
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
    # Sunucuya grafik arayüzü istemeyen "headless" sürümü zorla enjekte et
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.9.0.80"])
    import cv2

# Ana kütüphaneleri şimdi güvenle çağırabiliriz
import streamlit as st
import numpy as np
import time
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Proctoring Fixed", layout="wide")

st.title("🛡️ AKILLI SINAV GÜVENLİK SİSTEMİ (BULUT MODU)")
st.caption("Kendi Kendini Onaran Altyapı ve Arka Plan Telemetri Kaydı")

# Klasör kontrolü
if not os.path.exists("kopya_kanitlari"):
    os.makedirs("kopya_kanitlari")

# Raporlama için session state hafızası
if "final_rapor_verisi" not in st.session_state:
    st.session_state.final_rapor_verisi = []
if "sinav_bitti" not in st.session_state:
    st.session_state.sinav_bitti = False

# --- YAPAY ZEKA İŞLEMCİ SINIFI ---
mp_face_mesh = mp.solutions.face_mesh

class SmoothProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.durum = "BAŞLATILIYOR..."
        self.violation_score = 0.0
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        self.snapshot_saved = False
        self.init_time = time.time()
        self.history = []

    def recv(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape

        gecen_hazirlik_suresi = time.time() - self.init_time
        kalan_hazirlik = max(0.0, 3.0 - gecen_hazirlik_suresi)

        rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        sonuclar = self.face_mesh.process(rgb_kare)

        ihlal_var = False
        yaw, pitch = 0.0, 0.0

        if sonuclar.multi_face_landmarks:
            yuz_noktalari = sonuclar.multi_face_landmarks[0].landmark

            burun = np.array([yuz_noktalari[1].x * w, yuz_noktalari[1].y * h])
            sol_goz = np.array([yuz_noktalari[33].x * w, yuz_noktalari[33].y * h])
            sag_goz = np.array([yuz_noktalari[263].x * w, yuz_noktalari[263].y * h])
            alin = np.array([yuz_noktalari[10].x * w, yuz_noktalari[10].y * h])
            cene = np.array([yuz_noktalari[152].x * w, yuz_noktalari[152].y * h])

            sol_mesafe = np.linalg.norm(burun - sol_goz)
            sag_mesafe = np.linalg.norm(burun - sag_goz)
            yaw = float((sol_mesafe / (sag_mesafe + 1e-6) - 1.0) * 100.0)

            ust_mesafe = np.linalg.norm(burun - alin)
            alt_mesafe = np.linalg.norm(burun - cene)
            pitch = float((ust_mesafe / (alt_mesafe + 1e-6) - 1.2) * 100.0)

            self.last_yaw = yaw
            self.last_pitch = pitch

            if abs(yaw) > 25.0 or abs(pitch) > 20.0:
                ihlal_var = True

            if kalan_hazirlik == 0:
                cv2.line(kare, tuple(sol_goz.astype(int)), tuple(sag_goz.astype(int)), (255, 255, 0), 1)
                cv2.line(kare, tuple(alin.astype(int)), tuple(cene.astype(int)), (255, 255, 0), 1)
                cv2.circle(kare, tuple(burun.astype(int)), 4, (0, 255, 255), -1)
        else:
            if kalan_hazirlik == 0:
                ihlal_var = True

        if kalan_hazirlik > 0:
            self.durum = f"HAZIRLANIN ({kalan_hazirlik:.1f}s)"
            renk = (255, 140, 0)
        else:
            if self.durum.startswith("HAZIRLANIN"):
                self.durum = "SAFE (GÜVENLİ)"

            if self.durum != "SISTEM KILITLENDI":
                if ihlal_var:
                    self.durum = "UYARI: SÜPHELİ HAREKET"
                    self.violation_score = min(100.0, self.violation_score + 3.0)
                    if self.violation_score >= 100.0:
                        self.durum = "SISTEM KILITLENDI"
                else:
                    self.violation_score = max(0.0, self.violation_score - 1.0)
                    if self.violation_score == 0.0:
                        self.durum = "SAFE (GÜVENLİ)"

            if self.durum.startswith("SAFE"):
                renk = (0, 255, 0)
            elif self.durum.startswith("UYARI"):
                renk = (0, 255, 255)
            else:
                renk = (0, 0, 255)

        self.history.append({
            "Saniye": len(self.history) * 0.03,
            "Sağa/Sola Sapma (Yaw)": self.last_yaw,
            "Yukarı/Aşağı Sapma (Pitch)": self.last_pitch,
            "Anlık Risk Skoru": self.violation_score
        })

        cv2.rectangle(kare, (10, 10), (420, 140), (0, 0, 0), -1)
        cv2.putText(kare, f"DURUM: {self.durum}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        cv2.putText(kare, f"Yaw (Sag/Sol)  : {self.last_yaw:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(kare, f"Pitch (Ust/Alt) : {self.last_pitch:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(kare, (20, 115), (400, 125), (50, 50, 50), -1)
        cv2.rectangle(kare, (20, 115), (20 + int(self.violation_score * 3.8), 125), renk, -1)

        if self.durum == "SISTEM KILITLENDI" and not self.snapshot_saved:
            cv2.imwrite(f"kopya_kanitlari/ihlal_{int(time.time())}.jpg", kare)
            self.snapshot_saved = True

        if self.durum == "SISTEM KILITLENDI":
            cv2.rectangle(kare, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(kare, "KOPYA TESPITI: SINAV IPTAL!", (w // 2 - 220, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        return frame.from_ndarray(kare, format="bgr24")

# --- KULLANICI ARAYÜZÜ ---
if not st.session_state.sinav_bitti:
    sol_kolon, sag_kolon = st.columns([2, 1])

    with sol_kolon:
        st.markdown("### 📷 Canlı Kamera Denetimi")
        ctx = webrtc_streamer(
            key="smooth_project_stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SmoothProctorProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    with sag_kolon:
        st.markdown("### 🛠️ Sistem Kontrolleri")
        st.success("Sistem şuan Akıcı Performans Modunda çalışıyor.")
        
        if ctx.video_processor:
            if ctx.video_processor.durum == "SISTEM KILITLENDI":
                st.session_state.final_rapor_verisi = ctx.video_processor.history
                st.session_state.sinav_bitti = True
                st.rerun()

        if st.button("Sınavı Başarıyla Bitir ve Rapor Üret", type="primary"):
            if ctx.video_processor:
                st.session_state.final_rapor_verisi = ctx.video_processor.history
            st.session_state.sinav_bitti = True
            st.rerun()

else:
    st.markdown("## 📊 Sınav Değerlendirme ve Analiz Raporu")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Sınav Süresince Oluşan Hareket Grafiği")
        if st.session_state.final_rapor_verisi:
            df_rapor = pd.DataFrame(st.session_state.final_rapor_verisi).set_index("Saniye")
            st.line_chart(df_rapor)
        else:
            st.info("Grafik verisi bulunamadı.")
            
    with c2:
        st.markdown("#### 📂 Kanıt Odası (Snapshot)")
        if os.path.exists("kopya_kanitlari"):
            resimler = [f for f in os.listdir("kopya_kanitlari") if f.endswith(".jpg")]
            if resimler:
                en_yeni_resim = max([os.path.join("kopya_kanitlari", f) for f in resimler], key=os.path.getctime)
                st.image(en_yeni_resim, caption="Sistemin Otomatik Kaydettiği İhlal Anı", use_container_width=True)
            else:
                st.success("Temiz Sınav: Herhangi bir kural ihlali yaşanmadı.")

    if st.button("Yeniden Başlat"):
        st.session_state.final_rapor_verisi = []
        st.session_state.sinav_bitti = False
        st.rerun()
