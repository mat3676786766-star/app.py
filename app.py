import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Proctoring Matrix", layout="wide")

st.title("🛡️ PROFESSIONAL AI PROCTORING SYSTEM")
st.caption("MediaPipe Face Mesh ve Dinamik Ceza Puanı Tabanlı Kopya Analiz Modülü")

# --- MEDIAPIPE YÜZ YAPILANDIRMASI ---
mp_face_mesh = mp.solutions.face_mesh

class AdvancedProctorProcessor(VideoProcessorBase):
    def __init__(self):
        # MediaPipe işlemcisini başlatıyoruz
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.durum = "SAFE"
        self.violation_score = 0.0  # 0 ile 100 arasında ceza havuzu
        self.last_yaw = 0.0
        self.last_pitch = 0.0

    def recv(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape

        # Resmi RGB formatına çevir (MediaPipe RGB çalışır)
        rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        sonuclar = self.face_mesh.process(rgb_kare)

        ihlal_var = False
        yaw, pitch = 0.0, 0.0

        if sonuclar.multi_face_landmarks:
            yuz_noktalari = sonuclar.multi_face_landmarks[0].landmark

            # Kritik Yüz Noktalarının Koordinatlarını Piksele Çeviriyoruz
            # 1: Burun ucu, 33: Sol göz dışı, 263: Sağ göz dışı, 10: Alın, 152: Çene
            burun = np.array([yuz_noktalari[1].x * w, yuz_noktalari[1].y * h])
            sol_goz = np.array([yuz_noktalari[33].x * w, yuz_noktalari[33].y * h])
            sag_goz = np.array([yuz_noktalari[263].x * w, yuz_noktalari[263].y * h])
            alin = np.array([yuz_noktalari[10].x * w, yuz_noktalari[10].y * h])
            cene = np.array([yuz_noktalari[152].x * w, yuz_noktalari[152].y * h])

            # --- GEOMETRİK EKSEN HESAPLAMA (YAW & PITCH) ---
            # Yatayda burnun gözlere olan mesafesinin oranı (Sağ/Sol dönüşü verir)
            sol_mesafe = np.linalg.norm(burun - sol_goz)
            sag_mesafe = np.linalg.norm(burun - sag_goz)
            # Normal merkez oranı 1.0 civarıdır. Sapmaları dereceye simüle ediyoruz
            yaw = float((sol_mesafe / (sag_mesafe + 1e-6) - 1.0) * 100.0)

            # Dikeyde burnun alın ve çeneye olan mesafesinin oranı (Yukarı/Aşağı dönüşü verir)
            ust_mesafe = np.linalg.norm(burun - alin)
            alt_mesafe = np.linalg.norm(burun - cene)
            pitch = float((ust_mesafe / (alt_mesafe + 1e-6) - 1.2) * 100.0)

            self.last_yaw = yaw
            self.last_pitch = pitch

            # --- GERÇEKÇİ EŞİK DEĞERLERİ KONTROLÜ ---
            # Kafasını bariz şekilde sağa/sola (> 25) veya yukarı/aşağı (> 20) çevirdi mi?
            if abs(yaw) > 25.0 or abs(pitch) > 20.0:
                ihlal_var = True

            # Yüzün etrafına ince geometrik çizgiler çiz (Görsel geri bildirim)
            cv2.line(kare, tuple(sol_goz.astype(int)), tuple(sag_goz.astype(int)), (255, 255, 0), 1)
            cv2.line(kare, tuple(alin.astype(int)), tuple(cene.astype(int)), (255, 255, 0), 1)
            cv2.circle(kare, tuple(burun.astype(int)), 4, (0, 255, 255), -1)
        else:
            # Eğer ekranda hiç yüz yoksa bu da bir ihlaldir!
            ihlal_var = True

        # --- DİNAMİK CEZA PUANI MANTIĞI ---
        if self.durum != "SISTEM KILITLENDI":
            if ihlal_var:
                self.durum = "UYARI: SÜPHELİ HAREKET"
                # İhlal devam ettikçe ceza puanını kare başına 2.5 puan artır (~1.5 saniyede dolar)
                self.violation_score = min(100.0, self.violation_score + 2.5)
                if self.violation_score >= 100.0:
                    self.durum = "SISTEM KILITLENDI"
            else:
                # Kurallara uyuyorsa ceza puanını yavaşça eriterek affet
                self.violation_score = max(0.0, self.violation_score - 1.0)
                if self.violation_score == 0.0:
                    self.durum = "SAFE (GÜVENLİ)"

        # --- HUD (EKRAN ÜSTÜ) PANEL TASARIMI ---
        # Duruma göre renk seçimi
        if self.durum.startswith("SAFE"):
            renk = (0, 255, 0)       # Yeşil
        elif self.durum.startswith("UYARI"):
            renk = (0, 255, 255)     # Sarı
        else:
            renk = (0, 0, 255)       # Kırmızı

        # Siyah bilgi çubuğu arka planı
        cv2.rectangle(kare, (10, 10), (420, 140), (0, 0, 0), -1)
        
        # Metinleri ekrana bas
        cv2.putText(kare, f"DURUM: {self.durum}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        cv2.putText(kare, f"Yaw (Sag/Sol)  : {self.last_yaw:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(kare, f"Pitch (Ust/Alt) : {self.last_pitch:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ceza Barını Çiz (Görsel İlerleme Çubuğu)
        cv2.putText(kare, f"Risk Orani: %{int(self.violation_score)}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(kare, (160, 115), (400, 128), (50, 50, 50), -1) # Bar arka planı
        cv2.rectangle(kare, (160, 115), (160 + int(self.violation_score * 2.4), 128), renk, -1) # Doluluk oranı

        # Büyük Kilit Ekranı
        if self.durum == "SISTEM KILITLENDI":
            cv2.rectangle(kare, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(kare, "KOPYA TESPITI: SINAV IPTAL!", (w // 2 - 220, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        return frame.from_ndarray(kare, format="bgr24")

# WebRTC Başlatıcı
webrtc_streamer(
    key="advanced_proctor_mesh",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=AdvancedProctorProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
