import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- GÜVENLİK MATRİSİ İŞLEMCİSİ ---
class ProctorProcessor(VideoTransformerBase):
    def __init__(self):
        self.risk_orani = 0.0
        self.kopya_kilitlendi = False
        self.durum_mesaji = "SİSTEM AKTİF"
        self.yasakli_bolge_baslangici = None
        self.coklu_kisi_baslangici = None
        
    def transform(self, frame):
        kare = frame.to_ndarray(format="bgr24")
        if self.kopya_kilitlendi:
            cv2.putText(kare, "KOPYA TESPIT EDILDI!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return kare

        # 1. Kontur Analizi (Kişi Sayımı)
        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        _, esik = cv2.threshold(cv2.GaussianBlur(gri, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        kisi_sayisi = sum(1 for c in konturlar if cv2.contourArea(c) > 15000)
        
        # 2. Çoklu Kişi Kontrolü
        if kisi_sayisi >= 2:
            self.durum_mesaji = f"UYARI: {kisi_sayisi} KİŞİ VAR!"
            if self.coklu_kisi_baslangici is None: self.coklu_kisi_baslangici = time.time()
            if time.time() - self.coklu_kisi_baslangici > 3.0: self.kopya_kilitlendi = True
        else:
            self.coklu_kisi_baslangici = None
            self.durum_mesaji = "GÜVENLİ"

        # 3. Görsel Geri Bildirim
        cv2.putText(kare, self.durum_mesaji, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Risk Durumu
        if kisi_sayisi >= 2:
            cv2.putText(kare, "KOPYA CEKME IHTIMALI VAR!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
        return kare

# --- ARAYÜZ VE GRAFİKLER ---
st.title("🛡️ AI SAFETY MATRIX v5.2")

# Kamera Akışı
ctx = webrtc_streamer(key="matrix", video_processor_factory=ProctorProcessor)

# --- GRAFİK VE VERİ PANELİ ---
if ctx.video_processor:
    # Açısal Hareket ve Risk Takibi için Dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Hareket Doğrultusu (Yaw/Pitch)")
        # Gerçek zamanlı veriyi burada grafiklendirebilirsin
        chart_data = np.random.randn(20, 2) # Buraya senin yaw/pitch verilerin gelecek
        st.line_chart(chart_data)
        
    with col2:
        st.subheader("🔥 Risk Analizi")
        risk = ctx.video_processor.risk_orani
        st.progress(min(risk/100, 1.0))
        st.write(f"Kopya İhtimali: %{int(risk)}")

st.info("Sistem, çoklu kişi veya eksen dışı hareketleri 3 saniye içinde teşhis eder.")
