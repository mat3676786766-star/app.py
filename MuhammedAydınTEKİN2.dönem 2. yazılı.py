import streamlit as st
import cv2
import numpy as np
import time
import random
from datetime import datetime

# Sayfa Genişlik Ayarları
st.set_page_config(page_title="AI Safety Matrix v5.2", layout="wide")

FILTRE_BOYUTU = 5          
SINAV_SURESI_DAKIKA = 10  
TOLERANS_SURESI = 3.0

# Log Fonksiyonu
if "loglar" not in st.session_state:
    st.session_state.loglar = ["[SYSTEM ENTERING SMOOTH TOLERANCE MATRIX (v5.2)]"]

def web_log_kaydet(mesaj):
    zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.loglar.append(f"[{zaman_damgasi}] {mesaj}")

# Oturum Değişkenleri
if "sistem_baslatildi" not in st.session_state:
    st.session_state.update({
        "hareket_baslangic": None, "ekrandan_ayrilma": None, "coklu_kisi_baslangic": None,
        "durum_mesaji": "SISTEM HAZIRLANIYOR...", "renk": (0, 255, 0), "kopya_kilitlendi": False,
        "yaw_gecmisi": [], "pitch_gecmisi": [], "kalibrasyon_bitti": False, "kalibrasyon_baslangic": time.time(),
        "cal_yaw_offset": 0, "cal_pitch_offset": 0, "cal_yaw_list": [], "cal_pitch_list": [],
        "son_kontrol_zamani": time.time(), "kontrol_aktif": False, "kontrol_bekleme_suresi": random.randint(15, 25),
        "onay_suresi_baslangic": None, "sinav_baslangic_zamani": time.time(), "son_ortam_parlakligi": None,
        "isik_ihlal_baslangici": None, "sabotaj_baslangici": None, "yasakli_bolge_baslangici": None,
        "arka_plan_ihlal_baslangici": None, "risk_orani": 0.0,
        "isik_durumu_metni": "STABLE LIGHT", "images_arka_plan": None, "son_goz_kirpma_zamani": time.time(),
        "eski_gri_kare": None, "mikro_titresim_gecmisi": []
    })

st.title("🌐 CYBER DEFENSE - ULTIMATE AI SAFETY MATRIX (v5.2)")
st.caption("Yapay Zekâ Destekli Gelişmiş Sınav Güvenliği ve Prosedür Matrisi")

sol_kolon, sag_kolon = st.columns([3, 1])

with sol_kolon:
    st.markdown("### 📷 CANLI SENSÖR VE ANALİZ KAMERASI")
    kamera_butonu = st.toggle("Sistemi Güvenli Modda Başlat", value=False)
    kare_alani = st.empty()
    canlilik_butonu_alani = st.empty()

with sag_kolon:
    st.markdown("### 🖥️ PROCTOR MATRIX PANEL")
    st.markdown("---")
    kalan_sure_alani = st.empty()
    kitle_sayisi_alani = st.empty()
    sensor_durumu_alani = st.empty()
    risk_bar_alani = st.empty()
    st.markdown("#### 📜 SİSTEM GÜVENLİK LOGLARI")
    log_alani = st.empty()

# Canlılık butonu tıklandığında tetiklenecek fonksiyon
if st.session_state.kontrol_aktif:
    kalan_onay = max(0.0, 8.0 - (time.time() - st.session_state.onay_suresi_baslangic))
    if kalan_onay > 0:
        if canlilik_butonu_alani.button(f"🔴 CANLILIK DOĞRULAMASI: BURAYA TIKLAYIN ({kalan_onay:.1f}sn)", key="canlilik_dogrulama_butonu"):
            st.session_state.kontrol_aktif = False
            st.session_state.son_kontrol_zamani = time.time()
            st.session_state.kontrol_bekleme_suresi = random.randint(20, 40)
            st.session_state.risk_orani = max(0.0, st.session_state.risk_orani - 25.0)
            web_log_kaydet("CANLILIK TESTI BAŞARIYLA GEÇİLDİ.")
            st.rerun()
    else:
        st.session_state.kopya_kilitlendi = True
        web_log_kaydet("ALARM: Canlilik testi zaman asimi!")
        st.rerun()

if kamera_butonu and not st.session_state.kopya_kilitlendi:
    kamera = cv2.VideoCapture(0)
    
    # Döngüyü optimize etmek için Streamlit'in kendi akış yapısını simüle ediyoruz
    while kamera.isOpened() and not st.session_state.kopya_kilitlendi:
        ret, kare = kamera.read()
        if not ret or kare is None: 
            st.warning("Kameraya erişilemiyor. Lütfen izinleri kontrol edin.")
            break
        
        kare = cv2.flip(kare, 1)
        h, w, _ = kare.shape
        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        gri_bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
        ihlaller_tetiklendi = False
        
        # 1. MİKRO TİTREŞİM ANALİZİ
        if st.session_state.eski_gri_kare is not None and st.session_state.kalibrasyon_bitti:
            kare_farki = cv2.absdiff(gri_bulanik, st.session_state.eski_gri_kare)
            _, fark_esik = cv2.threshold(kare_farki, 20, 255, cv2.THRESH_BINARY)
            mikro_hareket = np.sum(fark_esik > 0)
            
            st.session_state.mikro_titresim_gecmisi.append(mikro_hareket)
            if len(st.session_state.mikro_titresim_gecmisi) > 30:
                st.session_state.mikro_titresim_gecmisi.pop(0)
                
            if 600 < mikro_hareket < 7500:
                st.session_state.son_goz_kirpma_zamani = time.time()
                
            if len(st.session_state.mikro_titresim_gecmisi) == 30:
                titresim_kararliligi = np.std(st.session_state.mikro_titresim_gecmisi)
                if titresim_kararliligi < 50.0 and mikro_hareket > 0:
                    st.session_state.isik_durumu_metni = "STREAM LOOP DETECTED"
                    st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 0.5)
                    ihlaller_tetiklendi = True

            if time.time() - st.session_state.son_goz_kirpma_zamani > 45.0:
                st.session_state.kopya_kilitlendi = True
                web_log_kaydet("ALARM: Goz kirpma dogrulamasi zaman asimi!")
                break
                
        st.session_state.eski_gri_kare = gri_bulanik.copy()

        # 2. SABOTAJ DETEKTÖRÜ
        anlik_parlaklik = np.mean(gri)
        if anlik_parlaklik < 15.0:
            st.session_state.isik_durumu_metni = "SABOTAGE DETECTED!"
            st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 1.5)
            ihlaller_tetiklendi = True
            if st.session_state.sabotaj_baslangici is None: st.session_state.sabotaj_baslangici = time.time()
            if time.time() - st.session_state.sabotaj_baslangici > TOLERANS_SURESI:
                st.session_state.kopya_kilitlendi = True
                web_log_kaydet("ALARM: Kamera sabote edildi!")
                break
        else:
            st.session_state.sabotaj_baslangici = None

        # 3. EKRAN IŞIĞI ANALİZİ
        if st.session_state.son_ortam_parlakligi is not None and anlik_parlaklik >= 15.0:
            parlaklik_degisimi = abs(anlik_parlaklik - st.session_state.son_ortam_parlakligi)
            if parlaklik_degisimi > 8.0:
                st.session_state.isik_durumu_metni = "FLASH DETECTED!"
                st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 2.0)
                ihlaller_tetiklendi = True
                if st.session_state.isik_ihlal_baslangici is None: st.session_state.isik_ihlal_baslangici = time.time()
                if time.time() - st.session_state.isik_ihlal_baslangici > TOLERANS_SURESI:
                    st.session_state.kopya_kilitlendi = True
                    web_log_kaydet("ALARM: Ekran/Telefon parlamasi!")
                    break
            else:
                if st.session_state.isik_ihlal_baslangici is not None and (time.time() - st.session_state.isik_ihlal_baslangici > 1.5):
                    st.session_state.isik_ihlal_baslangici = None
                    st.session_state.isik_durumu_metni = "STABLE LIGHT"
        if anlik_parlaklik >= 15.0: st.session_state.son_ortam_parlakligi = anlik_parlaklik

        # 4. YASAKLI BÖLGE VE KONTUR ANALİZİ
        yasakli_y_siniri = int(h * 0.75)
        cv2.line(kare, (0, yasakli_y_siniri), (w, yasakli_y_siniri), (0, 0, 255), 2)

        _, esik = cv2.threshold(gri_bulanik, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        en_buyuk_kontur = None
        maks_alan = 0
        gecerli_konturlar = []
        
        for kontur in konturlar:
            alan = cv2.contourArea(kontur)
            if alan > 12000: 
                gecerli_konturlar.append((alan, kontur))
                if alan > maks_alan:
                    maks_alan = alan
                    en_buyuk_kontur = kontur

        temiz_kitle_sayisi = 0
        for alan, kontur in gecerli_konturlar:
            if alan > (maks_alan * 0.45): temiz_kitle_sayisi += 1

        # 5. KALİBRASYON VE ODAK DÖNGÜLERİ
        if not st.session_state.kalibrasyon_bitti:
            st.session_state.durum_mesaji = f"KALIBRASYON MODU... {3.0 - (time.time() - st.session_state.kalibrasyon_baslangic):.1f}s"
            st.session_state.renk = (255, 255, 0)
            if temiz_kitle_sayisi == 1 and en_buyuk_kontur is not None:
                (x, y, w_box, h_box) = cv2.boundingRect(en_buyuk_kontur)
                st.session_state.cal_yaw_list.append(((x + w_box//2 - (w//2)) / (w//2)) * 45.0)
                st.session_state.cal_pitch_list.append(((y + h_box//2 - (h//2)) / (h//2)) * -35.0)
                
            if time.time() - st.session_state.kalibrasyon_baslangic > 3.0:
                if st.session_state.cal_yaw_list and st.session_state.cal_pitch_list:
                    st.session_state.cal_yaw_offset = sum(st.session_state.cal_yaw_list) / len(st.session_state.cal_yaw_list)
                    st.session_state.cal_pitch_offset = sum(st.session_state.cal_pitch_list) / len(st.session_state.cal_pitch_list)
                st.session_state.kalibrasyon_bitti = True
        else:
            if temiz_kitle_sayisi > 1:
                st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 1.0)
                ihlaller_tetiklendi = True
                if st.session_state.coklu_kisi_baslangic is None: st.session_state.coklu_kisi_baslangic = time.time()
                if time.time() - st.session_state.coklu_kisi_baslangic > TOLERANS_SURESI:
                    st.session_state.kopya_kilitlendi = True
                    web_log_kaydet("ALARM: Coklu kisi!")
                    break
                else:
                    st.session_state.durum_mesaji = f"UYARI: {temiz_kitle_sayisi} KITLE ALGILANDI!"
                    st.session_state.renk = (0, 165, 255)
            elif temiz_kitle_sayisi == 1 and en_buyuk_kontur is not None:
                (x, y, w_box, h_box) = cv2.boundingRect(en_buyuk_kontur)
                cv2.rectangle(kare, (x, y), (x + w_box, h_box + y), (255, 0, 0), 2)
                
                yuz_merkez_x = x + w_box // 2
                yuz_merkez_y = y + h_box // 2
                
                if yuz_merkez_y > yasakli_y_siniri:
                    st.session_state.isik_durumu_metni = "FORBIDDEN ZONE!"
                    st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 1.2)
                    ihlaller_tetiklendi = True
                    if st.session_state.yasakli_bolge_baslangici is None: st.session_state.yasakli_bolge_baslangici = time.time()
                    if time.time() - st.session_state.yasakli_bolge_baslangici > TOLERANS_SURESI:
                        st.session_state.kopya_kilitlendi = True
                        web_log_kaydet("ALARM: Yasakli alan ihlali!")
                        break
                else:
                    st.session_state.yasakli_bolge_baslangici = None

                # Açı Hesapları
                anlik_yaw = (((yuz_merkez_x - (w // 2)) / (w // 2)) * 45.0) - st.session_state.cal_yaw_offset
                anlik_pitch = (((yuz_merkez_y - (h // 2)) / (h // 2)) * -35.0) - st.session_state.cal_pitch_offset
                
                st.session_state.yaw_gecmisi.append(anlik_yaw)
                st.session_state.pitch_gecmisi.append(anlik_pitch)
                if len(st.session_state.yaw_gecmisi) > FILTRE_BOYUTU:
                    st.session_state.yaw_gecmisi.pop(0)
                    st.session_state.pitch_gecmisi.pop(0)
                    
                yaw_derece = sum(st.session_state.yaw_gecmisi) / len(st.session_state.yaw_gecmisi)
                pitch_derece = sum(st.session_state.pitch_gecmisi) / len(st.session_state.pitch_gecmisi)
                
                if abs(yaw_derece) > 16.0 or abs(pitch_derece) > 13.0:
                    st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 0.8)
                    ihlaller_tetiklendi = True
                    if st.session_state.hareket_baslangic is None: st.session_state.hareket_baslangic = time.time()
                    if time.time() - st.session_state.hareket_baslangic > TOLERANS_SURESI:
                        st.session_state.kopya_kilitlendi = True
                        web_log_kaydet("ALARM: Surekli eksen disi bakis!")
                        break
                    else:
                        st.session_state.durum_mesaji = "ACISAL LIMIT ASILDI!"
                        st.session_state.renk = (0, 165, 255)
                else:
                    st.session_state.hareket_baslangic = None
                    if not st.session_state.kontrol_aktif and yuz_merkez_y <= yasakli_y_siniri:
                        st.session_state.durum_mesaji = "ODAKLANDI (GUVENLI MATRIS)"
                        st.session_state.renk = (0, 255, 0)
                        
                cv2.putText(kare, f"Y: {yaw_derece:.1f} P: {pitch_derece:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, st.session_state.renk, 2)
            else:
                st.session_state.risk_orani = min(100.0, st.session_state.risk_orani + 1.5)
                ihlaller_tetiklendi = True
                if st.session_state.ekrandan_ayrilma is None: st.session_state.ekrandan_ayrilma = time.time()
                if time.time() - st.session_state.ekrandan_ayrilma > 4.0:
                    st.session_state.kopya_kilitlendi = True
                    web_log_kaydet("ALARM: Ekran tamamen terk edildi!")
                    break

        # Soğuma
        if not ihlaller_tetiklendi and st.session_state.kalibrasyon_bitti:
            st.session_state.risk_orani = max(0.0, st.session_state.risk_orani - 0.5)

        # Canlılık Testi Tetikleyici Kontrolü
        su_an = time.time()
        if not st.session_state.kontrol_aktif and (su_an - st.session_state.son_kontrol_zamani > st.session_state.kontrol_bekleme_suresi):
            st.session_state.kontrol_aktif = True
            st.session_state.onay_suresi_baslangic = time.time()
            st.rerun()

        # Sağ Panel Güncellemeleri
        gecen_sure = time.time() - st.session_state.sinav_baslangic_zamani
        toplam_saniye = max(0, (SINAV_SURESI_DAKIKA * 60) - gecen_sure)
        kalan_sure_alani.metric("⏱️ Kalan Sınav Süresi", f"{int(toplam_saniye // 60):02d}:{int(toplam_saniye % 60):02d}")
        kitle_sayisi_alani.text(f"Kitle Sayısı: {temiz_kitle_sayisi}")
        sensor_durumu_alani.code(f"Sistem Durumu: {st.session_state.durum_mesaji}\nSensör: {st.session_state.isik_durumu_metni}")
        
        risk_yuzde = int(st.session_state.risk_orani)
        risk_bar_alani.progress(risk_yuzde / 100.0, text=f"🔥 KOPYA İHTİMALİ: %{risk_yuzde}")
        log_alani.text_area("Canlı Akış", value="\n".join(st.session_state.loglar[-8:]), height=200)

        # Görüntüyü Bas
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_alani.image(kare_rgb, channels="RGB")
        time.sleep(0.05)
        
    kamera.release()

if st.session_state.kopya_kilitlendi:
    st.error("❌ MATRİS KİLİTLENDİ: GÜVENLİK İHLALİ TESPİT EDİLDİ!")
    risk_bar_alani.progress(1.0, text="🔥 KOPYA İHTİMALİ: %100 - TERMINATED")
    log_alani.text_area("Canlı Akış", value="\n".join(st.session_state.loglar[-8:]), height=200)