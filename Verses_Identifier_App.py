import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from difflib import SequenceMatcher
import json
import pyttsx3

# Initialize the model and processor
processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")

# Mapping of Surah names to JSON files
surah_files = {
    "Al-Fatiha (الفاتحة)": "surahs_json_files/001_al-fatiha.json",
    "Al-Baqarah (البقرة)": "surahs_json_files/002_al-baqarah.json",
    "Aal-E-Imran (آل عمران)": "surahs_json_files/003_aal-e-imran.json",
    "An-Nisa (النساء)": "surahs_json_files/004_an-nisa.json",
    "Al-Maidah (المائدة)": "surahs_json_files/005_al-maidah.json",
    "Al-An'am (الأنعام)": "surahs_json_files/006_al-anam.json",
    "Al-A'raf (الأعراف)": "surahs_json_files/007_al-a'raf.json",
    "Al-Anfal (الأنفال)": "surahs_json_files/008_al-anfal.json",
    "At-Tawbah (التوبة)": "surahs_json_files/009_at-tawbah.json",
    "Yunus (يونس)": "surahs_json_files/010_yunus.json",
    "Hud (هود)": "surahs_json_files/011_hud.json",
    "Yusuf (يوسف)": "surahs_json_files/012_yusuf.json",
    "Ar-Ra'd (الرعد)": "surahs_json_files/013_ar-rad.json",
    "Ibrahim (إبراهيم)": "surahs_json_files/014_ibrahim.json",
    "Al-Hijr (الحجر)": "surahs_json_files/015_al-hijr.json",
    "An-Nahl (النحل)": "surahs_json_files/016_an-nahl.json",
    "Al-Isra (الإسراء)": "surahs_json_files/017_al-isra.json",
    "Al-Kahf (الكهف)": "surahs_json_files/018_al-kahf.json",
    "Maryam (مريم)": "surahs_json_files/019_maryam.json",
    "Ta-Ha (طه)": "surahs_json_files/020_ta-ha.json",
    "Al-Anbiya (الأنبياء)": "surahs_json_files/021_al-anbiya.json",
    "Al-Hajj (الحج)": "surahs_json_files/022_al-hajj.json",
    "Al-Mu'minun (المؤمنون)": "surahs_json_files/023_al-mu'minun.json",
    "An-Nur (النور)": "surahs_json_files/024_an-nur.json",
    "Al-Furqan (الفرقان)": "surahs_json_files/025_al-furqan.json",
    "Ash-Shu'ara (الشعراء)": "surahs_json_files/026_ash-shu'ara.json",
    "An-Naml (النمل)": "surahs_json_files/027_an-naml.json",
    "Al-Qasas (القصص)": "surahs_json_files/028_al-qasas.json",
    "Al-Ankabut (العنكبوت)": "surahs_json_files/029_al-ankabut.json",
    "Ar-Rum (الروم)": "surahs_json_files/030_ar-rum.json",
    "Luqman (لقمان)": "surahs_json_files/031_luqman.json",
    "As-Sajdah (السجدة)": "surahs_json_files/032_as-sajdah.json",
    "Al-Ahzab (الأحزاب)": "surahs_json_files/033_al-ahzab.json",
    "Saba (سبأ)": "surahs_json_files/034_saba.json",
    "Fatir (فاطر)": "surahs_json_files/035_fatir.json",
    "Ya-Sin (يس)": "surahs_json_files/036_ya-sin.json",
    "As-Saffat (الصافات)": "surahs_json_files/037_as-saffat.json",
    "Sad (ص)": "surahs_json_files/038_sad.json",
    "Az-Zumar (الزمر)": "surahs_json_files/039_az-zumar.json",
    "Ghafir (غافر)": "surahs_json_files/040_ghafir.json",
    "Fussilat (فصلت)": "surahs_json_files/041_fussilat.json",
    "Ash-Shura (الشورى)": "surahs_json_files/042_ash-shura.json",
    "Az-Zukhruf (الزخرف)": "surahs_json_files/043_az-zukhruf.json",
    "Ad-Dukhkhan (الدخان)": "surahs_json_files/044_ad-dukhkhan.json",
    "Al-Jathiya (الجاثية)": "surahs_json_files/045_al-jathiya.json",
    "Al-Ahqaf (الأحقاف)": "surahs_json_files/046_al-ahqaf.json",
    "Muhammad (محمد)": "surahs_json_files/047_muhammad.json",
    "Al-Fath (الفتح)": "surahs_json_files/048_al-fath.json",
    "Al-Hujurat (الحجرات)": "surahs_json_files/049_al-hujurat.json",
    "Qaf (ق)": "surahs_json_files/050_qaf.json",
    "Adh-Dhariyat (الذاريات)": "surahs_json_files/051_adh-dhariyat.json",
    "At-Tur (الطور)": "surahs_json_files/052_at-tur.json",
    "An-Najm (النجم)": "surahs_json_files/053_an-najm.json",
    "Al-Qamar (القمر)": "surahs_json_files/054_al-qamar.json",
    "Ar-Rahman (الرحمن)": "surahs_json_files/055_ar-rahman.json",
    "Al-Waqi'ah (الواقعة)": "surahs_json_files/056_al-waqi'ah.json",
    "Al-Hadid (الحديد)": "surahs_json_files/057_al-hadid.json",
    "Al-Mujadila (المجادلة)": "surahs_json_files/058_al-mujadila.json",
    "Al-Hashr (الحشر)": "surahs_json_files/059_al-hashr.json",
    "Al-Mumtahina (الممتحنة)": "surahs_json_files/060_al-mumtahina.json",
    "As-Saff (الصف)": "surahs_json_files/061_as-saff.json",
    "Al-Jumu'ah (الجمعة)": "surahs_json_files/062_al-jumu'ah.json",
    "Al-Munafiqoon (المنافقون)": "surahs_json_files/063_al-munafiqoon.json",
    "At-Taghabun (التغابن)": "surahs_json_files/064_at-taghabun.json",
    "At-Talaq (الطلاق)": "surahs_json_files/065_at-talaq.json",
    "At-Tahrim (التحريم)": "surahs_json_files/066_at-tahrim.json",
    "Al-Mulk (الملك)": "surahs_json_files/067_al-mulk.json",
    "Al-Qalam (القلم)": "surahs_json_files/068_al-qalam.json",
    "Al-Haqqah (الحاقة)": "surahs_json_files/069_al-haqqah.json",
    "Al-Ma'arij (المعارج)": "surahs_json_files/070_al-ma'arij.json",
    "Nooh (نوح)": "surahs_json_files/071_nooh.json",
    "Al-Jinn (الجن)": "surahs_json_files/072_al-jinn.json",
    "Al-Muzzammil (المزمل)": "surahs_json_files/073_al-muzzammil.json",
    "Al-Muddathir (المدثر)": "surahs_json_files/074_al-muddathir.json",
    "Al-Qiyamah (القيامة)": "surahs_json_files/075_al-qiyamah.json",
    "Al-Insan (الإنسان)": "surahs_json_files/076_al-insan.json",
    "Al-Mursalat (المرسلات)": "surahs_json_files/077_al-mursalat.json",
    "An-Naba (النبأ)": "surahs_json_files/078_an-naba.json",
    "An-Nazi'at (النازعات)": "surahs_json_files/079_an-nazi'at.json",
    "Abasa (عبس)": "surahs_json_files/080_abasa.json",
    "At-Takwir (التكوير)": "surahs_json_files/081_at-takwir.json",
    "Al-Infitar (الإنفطار)": "surahs_json_files/082_al-infitar.json",
    "Al-Mutaffifin (المطففين)": "surahs_json_files/083_al-mutaffifin.json",
    "Al-Inshiqaq (الإنشقاق)": "surahs_json_files/084_al-inshiqaq.json",
    "Al-Buruj (البروج)": "surahs_json_files/085_al-buruj.json",
    "At-Tariq (الطارق)": "surahs_json_files/086_at-tariq.json",
    "Al-A'la (الأعلى)": "surahs_json_files/087_al-a'la.json",
    "Al-Ghashiyah (الغاشية)": "surahs_json_files/088_al-ghashiyah.json",
    "Al-Fajr (الفجر)": "surahs_json_files/089_al-fajr.json",
    "Al-Balad (البلد)": "surahs_json_files/090_al-balad.json",
    "Ash-Shams (الشمس)": "surahs_json_files/091_ash-shams.json",
    "Al-Lail (الليل)": "surahs_json_files/092_al-layl.json",
    "Ad-Duha (الضحى)": "surahs_json_files/093_ad-duha.json",
    "Ash-Sharh (الشرح)": "surahs_json_files/094_ash-sharh.json",
    "At-Tin (التين)": "surahs_json_files/095_at-tin.json",
    "Al-Alaq (العلق)": "surahs_json_files/096_al-'alaq.json",
    "Al-Qadr (القدر)": "surahs_json_files/097_al-qadr.json",
    "Al-Bayyina (البينة)": "surahs_json_files/098_al-bayyina.json",
    "Az-Zalzalah (الزلزلة)": "surahs_json_files/099_az-zalzalah.json",
    "Al-Adiyat (العاديات)": "surahs_json_files/100_al-adiyat.json",
    "Al-Qari'ah (القارعة)": "surahs_json_files/101_al-qari'ah.json",
    "At-Takathur (التكاثر)": "surahs_json_files/102_at-takathur.json",
    "Al-Asr (العصر)": "surahs_json_files/103_al-asr.json",
    "Al-Humazah (الهمزة)": "surahs_json_files/104_al-humazah.json",
    "Al-Fil (الفيل)": "surahs_json_files/105_al-fil.json",
    "Quraish (قريش)": "surahs_json_files/106_quraish.json",
    "Al-Ma'un (الماعون)": "surahs_json_files/107_al-ma'un.json",
    "Al-Kawthar (الكوثر)": "surahs_json_files/108_al-kawthar.json",
    "Al-Kafirun (الكافرون)": "surahs_json_files/109_al-kafirun.json",
    "An-Nasr (النصر)": "surahs_json_files/110_an-nasr.json",
    "Al-Masad (المسد)": "surahs_json_files/111_al-masad.json",
    "Al-Ikhlas (الإخلاص)": "surahs_json_files/112_al-ikhlas.json",
    "Al-Falaq (الفلق)": "surahs_json_files/113_al-falaq.json",
    "An-Nas (الناس)": "surahs_json_files/114_an-nas.json",
}

# Function to load all surahs with their verses
def load_all_surahs():
    quran_text_arabic = {}
    for surah_name, json_file in surah_files.items():
        with open(json_file, "r", encoding="utf-8") as file:
            quran_data = json.load(file)
            if 'ayahs' in quran_data and isinstance(quran_data['ayahs'], list):
                verses = [ayah['text'] for ayah in quran_data['ayahs']]
                quran_text_arabic[surah_name] = verses
            else:
                print(f"Unexpected structure in {json_file}")
    return quran_text_arabic

quran_text_arabic = load_all_surahs()

# Function to transcribe audio
def transcribe_audio(audio_chunk, sample_rate=16000):
    inputs = processor(audio_chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Function to calculate similarity between two texts
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to find the ayah(s) matching the transcription
def find_ayah_arabic(transcription, threshold=0.6):
    matches = []
    for surah, ayahs in quran_text_arabic.items():
        for ayah_num, ayah in enumerate(ayahs, start=1):
            similarity_score = calculate_similarity(transcription, ayah)
            if similarity_score >= threshold:
                matches.append((surah, ayah_num, similarity_score))
    if matches:
        return sorted(matches, key=lambda x: x[2], reverse=True)  # Return matches sorted by similarity score
    return []

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🔊 Quranic Ayah Identifier 🎤")
        self.root.state("zoomed")

        # Initialize the TTS engine
        self.tts_engine = pyttsx3.init()

        # Load and set background image
        self.background_image = Image.open("Background.png")
        self.bg_photo = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(root, image=self.bg_photo)
        self.background_label.place(relwidth=1, relheight=1)

        # Configure fonts
        self.title_font = tkFont.Font(family="Helvetica", size=24, weight="bold")
        self.button_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
        self.label_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

        # Custom style for rounded buttons using images
        style = ttk.Style()
        style.configure('Rounded.TButton', font=self.button_font, padding=10)
        style.map('Rounded.TButton', background=[('active', '#005A8B'), ('!active', '#007ACC')])

        # Title Label
        self.title_label = tk.Label(root, text="🔊 Quranic Ayah Identifier 🎤", font=self.title_font, bg='#ffffff', relief=tk.FLAT)
        self.title_label.pack(pady=20)

        # Frame for radio buttons
        frame_radio_buttons = tk.Frame(root, bg='#ffffff')
        frame_radio_buttons.pack(pady=10)

        # Recording Mode Options
        self.recording_mode = tk.StringVar(value="single")
        self.single_ayah_radio = tk.Radiobutton(frame_radio_buttons, text="Record One Ayah", variable=self.recording_mode, value="single", command=self.update_mode, indicatoron=0, width=20, height=1, relief=tk.RAISED, font=self.button_font, bg='#ffffff', selectcolor='#ff3333')
        self.multiple_ayah_radio = tk.Radiobutton(frame_radio_buttons, text="Record Multiple Ayahs", variable=self.recording_mode, value="multiple", command=self.update_mode, indicatoron=0, width=20, height=1, relief=tk.RAISED, font=self.button_font, bg='#ffffff', selectcolor='#ff3333')

        self.single_ayah_radio.pack(side=tk.LEFT, padx=10)
        self.multiple_ayah_radio.pack(side=tk.LEFT, padx=10)

        # Create and place widgets
        self.instructions_label = tk.Label(root, text="Press 'Start Recording' to begin.", font=self.label_font, bg='#ffffff', relief=tk.FLAT)
        self.instructions_label.pack(pady=10)

        self.start_button = ttk.Button(root, text="🎙 Start Recording", command=self.start_recording, style='Rounded.TButton')
        self.start_button.pack(pady=10)

        self.add_ayah_button = ttk.Button(root, text="➕ Add Ayah", command=self.add_ayah, state=tk.DISABLED, style='Rounded.TButton')
        self.add_ayah_button.pack(pady=10)
        self.add_ayah_button.pack_forget()

        self.stop_button = ttk.Button(root, text="🛑 Stop Recording", command=self.stop_recording, state=tk.DISABLED, style='Rounded.TButton')
        self.stop_button.pack(pady=10)

        # Transcript Title and Text
        self.transcription_title_label = tk.Label(root, text="Transcript:", font=self.label_font, bg='#ffffff', relief=tk.FLAT)
        self.transcription_title_label.pack(pady=10)

        self.transcription_label = tk.Label(root, text="", font=self.label_font, bg='#ffffff', justify='right', relief=tk.FLAT)
        self.transcription_label.pack(pady=10)

        # Ayah Title and Text
        self.ayah_title_label = tk.Label(root, text="Identified Ayah(s):", font=self.label_font, bg='#ffffff', relief=tk.FLAT)
        self.ayah_title_label.pack(pady=10)

        self.ayah_label = tk.Label(root, text="", font=self.label_font, bg='#ffffff', justify='left')
        self.ayah_label.pack(pady=10)

        # Initialize variables for recording
        self.is_recording = False
        self.audio_data = []
        self.current_ayah = 1
        self.transcriptions = []

    def update_mode(self):
        self.is_recording = False
        self.audio_data = []
        self.transcriptions = []
        self.current_ayah = 1
        self.transcription_label.config(text="")
        self.ayah_label.config(text="")
        self.instructions_label.config(text="Press 'Start Recording' to begin.")

        # Update button state based on selected mode
        if self.recording_mode.get() == "single":
            self.add_ayah_button.pack_forget()
        else:
            self.add_ayah_button.pack()

    def start_recording(self):
        self.update_mode()
        self.is_recording = True
        self.audio_data = []
        self.start_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL
        self.add_ayah_button['state'] = tk.NORMAL if self.recording_mode.get() == "multiple" else tk.DISABLED
        sd.default.samplerate = 16000
        sd.default.channels = 1
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()
        if self.recording_mode.get() == "multiple":
            self.instructions_label.config(text=f"Recording Ayah {self.current_ayah}")
        else:
            self.instructions_label.config(text="Recording...")

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def add_ayah(self):
        if self.is_recording and self.recording_mode.get() == "multiple":
            audio_array = np.concatenate(self.audio_data, axis=0)
            audio_tensor = torch.from_numpy(audio_array.flatten())

            transcription = transcribe_audio(audio_tensor)
            self.transcriptions.append(transcription)
            self.transcription_label.config(text="\n".join(self.transcriptions))

            self.audio_data = []
            self.current_ayah += 1
            self.instructions_label.config(text=f"Recording Ayah {self.current_ayah}")

    def stop_recording(self):
        self.is_recording = False
        self.start_button['state'] = tk.NORMAL
        self.stop_button['state'] = tk.DISABLED
        self.add_ayah_button['state'] = tk.DISABLED
        self.stream.stop()

        if self.recording_mode.get() == "single" and self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            audio_tensor = torch.from_numpy(audio_array.flatten())

            transcription = transcribe_audio(audio_tensor)
            self.transcription_label.config(text=f"{transcription}")

            best_match = find_ayah_arabic(transcription)
            if best_match:
                match_text = f"Surah: {best_match[0][0]}, Ayah: {best_match[0][1]}, Similarity: {best_match[0][2]:.2f}"
                self.ayah_label.config(text=match_text)
                self.tts_engine.say(match_text)
                self.tts_engine.runAndWait()
            else:
                self.ayah_label.config(text="No matching Ayah found.")
                self.tts_engine.say("No matching Ayah found.")
                self.tts_engine.runAndWait()

        elif self.recording_mode.get() == "multiple" and self.transcriptions:
            ayah_matches = []
            for i, transcription in enumerate(self.transcriptions, start=1):
                self.transcription_label.config(text=f"{i}) {transcription}")
                
                best_match = find_ayah_arabic(transcription)
                if best_match:
                    match_text = f"Surah: {best_match[0][0]}, Ayah: {best_match[0][1]}, Similarity: {best_match[0][2]:.2f}"
                    ayah_matches.append(match_text)
                    self.tts_engine.say(match_text)
                    self.tts_engine.runAndWait()

            if ayah_matches:
                self.ayah_label.config(text="\n".join(ayah_matches))
            else:
                self.ayah_label.config(text="No matching Ayah(s) found.")
                self.tts_engine.say("No matching Ayah(s) found.")
                self.tts_engine.runAndWait()

        self.instructions_label.config(text="Recording stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()