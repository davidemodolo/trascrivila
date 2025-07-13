import customtkinter as ctk
import threading
import queue
import time
import json
import os
import traceback
import subprocess
from datetime import datetime

from summarizer import summarize_with_ollama, summarize_with_openai  # Ensure this file exists

# --- Backend Imports ---
import numpy as np
import whisper
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import sounddevice as sd
import resampy

from pyannote.audio import Pipeline
import wave

# --- App Configuration & Globals ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
WHISPER_SAMPLE_RATE = 16000
IS_GPU_AVAILABLE = torch.cuda.is_available()
print(f"[INFO] CUDA available: {IS_GPU_AVAILABLE}")


class ConfirmationDialog(ctk.CTkToplevel):
    def __init__(self, parent, title, message):
        super().__init__(parent)
        self.title(title)
        self.geometry("350x150")
        self.transient(parent)
        self.grab_set()
        self.result = False

        self.label = ctk.CTkLabel(self, text=message, wraplength=320)
        self.label.pack(padx=20, pady=20, expand=True)

        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=(0, 20))

        self.ok_button = ctk.CTkButton(self.button_frame, text="OK", command=self.on_ok, width=100)
        self.ok_button.pack(side="left", padx=10)

        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.on_cancel, width=100)
        self.cancel_button.pack(side="right", padx=10)

        self.wait_window()

    def on_ok(self):
        self.result = True
        self.destroy()

    def on_cancel(self):
        self.result = False
        self.destroy()


# --- Main Application Class ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local Real-Time Transcriber")
        self.geometry("1100x800")

        self.is_recording = False
        self.is_paused = False
        self.app_running = True
        self.gui_queue = queue.Queue()
        self.mic_data_queue = queue.Queue()
        self.sys_data_queue = queue.Queue()
        self.capture_threads = {"mic": {"thread": None, "stop_event": threading.Event()}, "sys": {"thread": None, "stop_event": threading.Event()}}
        self.processor_threads = []
        self.last_interim_text = {"mic": "", "sys": ""}
        self.whisper_model = None
        self.current_model_name = ""
        self.whisper_lock = threading.Lock()

        self.combined_transcript = []
        self.recording_start_time = None
        self.sys_transcript_with_timestamps = [] # Still needed for post-processing

        self.diarization_pipeline = None
        self.sys_audio_for_diarization = []
        self.speaker_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_ui()

    def start_app_logic(self):
        self.after(50, self.process_gui_queue)
        self.populate_device_menus()
        self.load_whisper_model()
        self.load_diarization_pipeline()
        self.start_capture_threads()

    def setup_ui(self):
        self.grid_columnconfigure((0, 1), weight=1); self.grid_rowconfigure(2, weight=1)
        self.top_frame = ctk.CTkFrame(self, corner_radius=10); self.top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.top_frame.grid_columnconfigure(3, weight=1)

        self.toggle_button = ctk.CTkButton(self.top_frame, text="Start Recording", command=self.toggle_recording, width=150); self.toggle_button.grid(row=0, column=0, padx=10, pady=10)
        self.pause_button = ctk.CTkButton(self.top_frame, text="Pause", command=self.toggle_pause, state="disabled"); self.pause_button.grid(row=0, column=1, padx=(0,10), pady=10)

        self.language_menu = ctk.CTkOptionMenu(self.top_frame, values=["english", "italian", "german", "spanish", "french"], command=self.on_language_change); self.language_menu.set("english"); self.language_menu.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.status_label = ctk.CTkLabel(self.top_frame, text="Status: Initializing...", text_color="gray"); self.status_label.grid(row=0, column=3, padx=10, pady=10, sticky="e")
        self.device_frame = ctk.CTkFrame(self, corner_radius=10); self.device_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.device_frame.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(self.device_frame, text="Microphone:").grid(row=0, column=0, padx=(10,0))
        self.mic_device_menu = ctk.CTkOptionMenu(self.device_frame, values=["loading..."], command=self.on_mic_device_change); self.mic_device_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.device_frame, text="System Audio (Monitor):").grid(row=0, column=2, padx=(10,0))
        self.sys_device_menu = ctk.CTkOptionMenu(self.device_frame, values=["loading..."], command=self.on_sys_device_change); self.sys_device_menu.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(self.device_frame, text="Speakers:").grid(row=0, column=4, padx=(10,0))
        self.num_speakers_entry = ctk.CTkEntry(self.device_frame, width=50); self.num_speakers_entry.grid(row=0, column=5, padx=(0,10), pady=10)
        self.num_speakers_entry.insert(0, "2")

        self.create_transcription_frame("mic", 2, 0); self.create_transcription_frame("sys", 2, 1)
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=10); self.bottom_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.api_key_entry = ctk.CTkEntry(self.bottom_frame, placeholder_text="Enter OpenAI API Key (if Ollama fails)"); self.api_key_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.summary_button = ctk.CTkButton(self.bottom_frame, text="Generate Summary", command=self.generate_summary_threaded, state="disabled"); self.summary_button.grid(row=0, column=1, padx=10, pady=10)
        self.summary_textbox = ctk.CTkTextbox(self.bottom_frame, height=100, state="disabled", wrap="word"); self.summary_textbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.export_button = ctk.CTkButton(self.bottom_frame, text="Export Transcript", command=self.export_transcript, state="disabled"); self.export_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def create_transcription_frame(self, source_type, row, col):
        padx = (10, 5) if source_type == "mic" else (5, 10)
        title = "You (Microphone)" if source_type == "mic" else "Call Audio (System)"
        frame = ctk.CTkFrame(self, corner_radius=10); frame.grid(row=row, column=col, padx=padx, pady=(0, 10), sticky="nsew")
        frame.grid_rowconfigure(1, weight=1); frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=(5,0))
        textbox = ctk.CTkTextbox(frame, state="disabled", wrap="word"); textbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        volume_bar = ctk.CTkProgressBar(frame, orientation="horizontal"); volume_bar.set(0); volume_bar.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        if source_type == "mic": self.mic_textbox, self.mic_volume_bar = textbox, volume_bar
        else: self.sys_textbox, self.sys_volume_bar = textbox, volume_bar

    def on_closing(self):
        print("[INFO] Closing application..."); self.app_running = False
        if self.is_recording: self.stop_recording()
        time.sleep(0.5)
        print("[INFO] All threads signaled to stop. Exiting."); self.destroy()

    def update_status(self, text, color="gray"): self.gui_queue.put({"type": "status_update", "text": text, "color": color})

    def get_pulse_monitor_sources(self):
        try:
            result = subprocess.run(['pactl', 'list', 'short', 'sources'], stdout=subprocess.PIPE, text=True, check=True, timeout=2)
            return [line.split()[1] for line in result.stdout.strip().split('\n') if ".monitor" in line]
        except Exception: return []

    def populate_device_menus(self):
        try:
            sd._terminate(); sd._initialize()
            mic_devices = [f"{i}: {dev['name']}" for i, dev in enumerate(sd.query_devices()) if dev.get('max_input_channels', 0) > 0]
            if not mic_devices: mic_devices = ["No input devices found"]
            self.mic_device_menu.configure(values=mic_devices, state="normal" if mic_devices[0] != "No input devices found" else "disabled")
            default_mic_index = sd.default.device[0]
            mic_default = next((name for name in mic_devices if name.startswith(f"{default_mic_index}:")), mic_devices[0])
            self.mic_device_menu.set(mic_default)
        except Exception as e: print(f"[ERROR] Mic devices error: {e}"); self.mic_device_menu.configure(values=["Error"], state="disabled")
        
        sys_devices = self.get_pulse_monitor_sources()
        if not sys_devices: sys_devices = ["No monitor sources found"]
        self.sys_device_menu.configure(values=sys_devices, state="normal" if sys_devices[0] != "No monitor sources found" else "disabled")
        if sys_devices[0] != "No monitor sources found": self.sys_device_menu.set(sys_devices[0])

    def on_mic_device_change(self, choice): self.restart_capture_thread("mic")
    def on_sys_device_change(self, choice): self.restart_capture_thread("sys")
        
    def start_capture_threads(self):
        self.restart_capture_thread("mic"); self.restart_capture_thread("sys")

    def restart_capture_thread(self, source_name):
        stop_event = self.capture_threads[source_name]["stop_event"]
        stop_event.set()
        if self.capture_threads[source_name]["thread"] is not None and self.capture_threads[source_name]["thread"].is_alive():
            self.capture_threads[source_name]["thread"].join(timeout=1.0)
        
        stop_event.clear()
        data_queue = self.mic_data_queue if source_name == "mic" else self.sys_data_queue
        thread = None
        if source_name == 'mic':
            device_name = self.mic_device_menu.get()
            if "No input" in device_name or "Error" in device_name: return
            try: device_index = int(device_name.split(':', 1)[0])
            except (ValueError, IndexError): return
            thread = threading.Thread(target=self.sounddevice_capture_thread, args=(device_index, data_queue, stop_event), daemon=True)
        else:
            device_name = self.sys_device_menu.get()
            if "No monitor" in device_name: return
            thread = threading.Thread(target=self.parec_capture_thread, args=(device_name, data_queue, stop_event), daemon=True)
        
        if thread:
            thread.start()
            self.capture_threads[source_name]["thread"] = thread

    def sounddevice_capture_thread(self, device_index, data_queue, stop_event):
        source_name = "mic"
        try:
            dev_info = sd.query_devices(device_index)
            native_rate = int(dev_info['default_samplerate'])
            chunk_size = 1024
            print(f"[{source_name.upper()}] Opening sounddevice stream on '{dev_info['name']}' at {native_rate}Hz...")

            with sd.InputStream(device=device_index, channels=1, samplerate=native_rate, dtype='float32', blocksize=chunk_size) as stream:
                print(f"[{source_name.upper()}] Stream opened successfully.")
                while not stop_event.is_set():
                    indata, overflowed = stream.read(chunk_size)
                    if overflowed: print(f"[WARN] Mic stream overflowed!")
                    self.gui_queue.put({"type": "volume_update", "source": "mic", "volume": np.linalg.norm(indata) * 10})
                    if self.is_recording:
                        resampled_data = resampy.resample(indata.flatten(), native_rate, WHISPER_SAMPLE_RATE)
                        data_queue.put(resampled_data)
        except Exception as e:
            print(f"[ERROR] Mic capture failed: {e}"); traceback.print_exc()
        print(f"[{source_name.upper()}] Capture thread stopped.")

    def parec_capture_thread(self, device_name, data_queue, stop_event):
        source_name = "sys"; parec = None; PAREC_RATE = 44100; CHUNK = 2048
        try:
            command = ["parec", f"--device={device_name}", "--format=s16le", f"--rate={PAREC_RATE}", "--channels=1"]
            parec = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[INFO] System audio stream opened for '{device_name}'")
            while not stop_event.is_set():
                raw = parec.stdout.read(CHUNK)
                if not raw: time.sleep(0.01); continue
                audio_np_s16 = np.frombuffer(raw, dtype=np.int16)
                audio_np_f32 = audio_np_s16.astype(np.float32) / 32768.0
                self.gui_queue.put({"type": "volume_update", "source": "sys", "volume": np.linalg.norm(audio_np_f32)*10})
                if self.is_recording and audio_np_f32.shape[0] > 1:
                    data_queue.put(resampy.resample(audio_np_f32, PAREC_RATE, WHISPER_SAMPLE_RATE))
        except Exception as e: print(f"[ERROR] System audio capture failed: {e}")
        finally:
            if parec: parec.terminate(); parec.wait()
        print(f"[{source_name.upper()}] Capture thread stopped.")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            mic_text = self.mic_textbox.get("1.0", "end-1c").strip()
            sys_text = self.sys_textbox.get("1.0", "end-1c").strip()
            if mic_text or sys_text:
                dialog = ConfirmationDialog(self, "Confirm", "Starting a new recording will clear the current transcript. Continue?")
                if dialog.result:
                    self.start_recording()
            else:
                self.start_recording()

    def toggle_pause(self):
        if not self.is_recording: return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.update_status("Status: Paused", "orange")
            self.pause_button.configure(text="Resume")
        else:
            self.update_status("Status: Recording...", "green")
            self.pause_button.configure(text="Pause")

    def start_recording(self):
        if self.whisper_model is None:
            self.update_status("Error: Whisper model not loaded.", "red"); return
        
        self.combined_transcript = []
        self.recording_start_time = time.time()
        self.sys_transcript_with_timestamps = []
        self.sys_audio_for_diarization = []
        self.last_interim_text = {"mic": "", "sys": ""}

        for textbox in [self.mic_textbox, self.sys_textbox]:
            textbox.configure(state="normal"); textbox.delete("1.0", "end"); textbox.configure(state="disabled")
        
        self.toggle_button.configure(text="Stop Recording", fg_color="#DB4437", hover_color="#C53727")
        self.pause_button.configure(state="normal", text="Pause")
        for widget in [self.summary_button, self.export_button, self.language_menu, self.mic_device_menu, self.sys_device_menu, self.num_speakers_entry]:
            widget.configure(state="disabled")
        
        self.is_paused = False
        self.is_recording = True

        mic_processor = threading.Thread(target=self.audio_processor_thread, args=("mic", self.mic_data_queue), daemon=True)
        sys_processor = threading.Thread(target=self.audio_processor_thread, args=("sys", self.sys_data_queue), daemon=True)
        self.processor_threads = [mic_processor, sys_processor]
        mic_processor.start(); sys_processor.start()
        
        self.update_status("Status: Recording...", "green")

    def stop_recording(self):
        self.is_recording = False
        self.is_paused = False
        self.update_status("Status: Finalizing...", "orange")

    def on_recording_finished(self):
        if self.diarization_pipeline and self.sys_transcript_with_timestamps:
            self.update_status("Status: Diarizing...", "orange")
            self.pause_button.configure(state="disabled", text="Pause")
            self.toggle_button.configure(state="disabled")
            threading.Thread(target=self.run_diarization, daemon=True).start()
        else:
            self.gui_queue.put({"type": "diarization_finished"})

    def on_language_change(self, choice):
        new_model_name = "base.en" if choice == "english" else "base"
        if self.current_model_name != new_model_name:
            self.load_whisper_model()

    def load_whisper_model(self):
        lang = self.language_menu.get()
        model_name = "base.en" if lang == "english" else "base"
        
        if self.whisper_model and self.current_model_name == model_name:
            print(f"[INFO] Model '{model_name}' is already loaded.")
            return

        self.update_status(f"Loading {model_name} model...", "orange")
        def load():
            try:
                device = "cuda" if IS_GPU_AVAILABLE else "cpu"
                self.whisper_model = whisper.load_model(model_name, device=device)
                self.current_model_name = model_name
                self.update_status("Status: Ready", "gray")
                print(f"[INFO] Model '{model_name}' loaded on {device}.")
            except Exception as e:
                self.update_status("Error loading model!", "red"); print(f"[ERROR] Model load fail: {e}")
                self.whisper_model = None; self.current_model_name = ""
        threading.Thread(target=load, daemon=True).start()

    def load_diarization_pipeline(self):
        self.update_status("Loading diarization model...", "orange")
        def load():
            try:
                self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                if IS_GPU_AVAILABLE:
                    self.diarization_pipeline.to(torch.device("cuda"))
                self.update_status("Status: Ready", "gray")
                print("[INFO] Diarization pipeline loaded.")
            except Exception as e:
                self.update_status("Diarization model failed to load", "red")
                print(f"[ERROR] Diarization pipeline failed to load: {e}\n[INFO] Make sure you have accepted the user conditions on Hugging Face and logged in via `huggingface-cli login`.")
        threading.Thread(target=load, daemon=True).start()


    def process_gui_queue(self):
        try:
            while not self.gui_queue.empty():
                msg = self.gui_queue.get_nowait()
                msg_type = msg.get("type")
                if msg_type == "transcript_update":
                    source, text, is_final = msg["source"], msg["text"], msg["is_final"]
                    textbox = self.mic_textbox if source == "mic" else self.sys_textbox
                    textbox.configure(state="normal")
                    if self.last_interim_text[source]:
                        start_index = textbox.index(f"end-{len(self.last_interim_text[source])+1}c")
                        textbox.delete(start_index, "end")
                    textbox.insert("end", text)
                    if is_final:
                        textbox.insert("end", "\n\n")
                        self.last_interim_text[source] = ""
                    else:
                        self.last_interim_text[source] = text
                    textbox.configure(state="disabled"); textbox.yview_moveto(1.0)

                elif msg_type == "volume_update":
                    if msg["source"] == "mic": self.mic_volume_bar.set(msg["volume"])
                    else: self.sys_volume_bar.set(msg["volume"])
                elif msg_type == "processors_finished": self.on_recording_finished()
                elif msg_type == "status_update": self.status_label.configure(text=msg["text"], text_color=msg["color"])
                elif msg_type == "summary_result":
                    self.summary_textbox.configure(state="normal"); self.summary_textbox.delete("1.0", "end"); self.summary_textbox.insert("1.0", msg.get("data", "No summary returned.")); self.summary_textbox.configure(state="disabled")
                    self.summary_button.configure(state="normal", text="Generate Summary")
                elif msg_type == "diarization_finished":
                    self.update_status("Status: Idle", "gray")
                    self.toggle_button.configure(state="normal", text="Start Recording", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
                    for widget in [self.summary_button, self.export_button, self.language_menu, self.mic_device_menu, self.sys_device_menu, self.num_speakers_entry]:
                         if widget not in (self.mic_device_menu, self.sys_device_menu) or "No" not in widget.get():
                            widget.configure(state="normal")
                    print("[INFO] Recording and Diarization session finished.")
                elif msg_type == "diarization_update":
                    # This message type now receives a full phrase
                    self.sys_textbox.configure(state="normal")
                    self.sys_textbox.insert("end", msg['text'] + "\n\n", (f"speaker_{msg['speaker_idx']}",))
                    self.sys_textbox.configure(state="disabled")
                    self.sys_textbox.yview_moveto(1.0)
        except queue.Empty: pass
        finally: self.after(50, self.process_gui_queue)

    def audio_processor_thread(self, source_name, data_queue):
        INTERIM_TIMEOUT = 0.7; PHRASE_TIMEOUT = 2.5
        audio_buffer = np.array([], dtype=np.float32)
        last_audio_time = last_interim_time = time.time()
        
        print(f"[{source_name.upper()}] Processor thread started.")
        
        while self.is_recording:
            if self.is_paused:
                time.sleep(0.1); continue

            try:
                now = time.time()
                try:
                    chunk = data_queue.get(timeout=0.5)
                    audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                    last_audio_time = now
                    while True:
                        chunk = data_queue.get_nowait()
                        audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                        last_audio_time = now
                except queue.Empty: pass

                time_since_last_audio = now - last_audio_time
                time_since_interim = now - last_interim_time
                
                is_final = time_since_last_audio > PHRASE_TIMEOUT
                
                if (time_since_interim > INTERIM_TIMEOUT or is_final) and audio_buffer.size > 1000:
                    lang = self.language_menu.get() if self.language_menu.get() != "english" else None
                    with self.whisper_lock:
                        result = self.whisper_model.transcribe(audio_buffer, language=lang, fp16=IS_GPU_AVAILABLE, word_timestamps=True)
                    text = result['text'].strip()

                    if text:
                        self.gui_queue.put({"type": "transcript_update", "source": source_name, "text": text, "is_final": is_final})
                        if is_final:
                            # --- MODIFIED: Add to combined transcript data ---
                            if source_name == 'mic':
                                buffer_duration = audio_buffer.size / WHISPER_SAMPLE_RATE
                                buffer_start_time = time.time() - buffer_duration
                                for segment in result['segments']:
                                    self.combined_transcript.append({
                                        "start": buffer_start_time + segment['start'],
                                        "speaker": "mic",
                                        "text": segment['text'].strip()
                                    })
                            elif source_name == 'sys':
                                self.sys_transcript_with_timestamps.extend(result['segments'])
                                self.sys_audio_for_diarization.append(audio_buffer.copy())

                    last_interim_time = now

                if is_final:
                    audio_buffer = np.array([], dtype=np.float32)
                    self.last_interim_text[source_name] = ""

                time.sleep(0.1)
            except Exception as e: print(f"[{source_name.upper()}] Processor error: {e}"); traceback.print_exc(); break

        while not data_queue.empty():
            try: audio_buffer = np.concatenate((audio_buffer, data_queue.get_nowait().flatten()))
            except queue.Empty: break
        
        if audio_buffer.size > 1000:
            lang = self.language_menu.get() if self.language_menu.get() != "english" else None
            with self.whisper_lock:
                result = self.whisper_model.transcribe(audio_buffer, language=lang, fp16=IS_GPU_AVAILABLE, word_timestamps=True)
            text = result['text'].strip()
            if text:
                self.gui_queue.put({"type": "transcript_update", "source": source_name, "text": text, "is_final": True})
                if source_name == 'mic':
                    buffer_duration = audio_buffer.size / WHISPER_SAMPLE_RATE
                    buffer_start_time = time.time() - buffer_duration
                    for segment in result['segments']:
                        self.combined_transcript.append({
                            "start": buffer_start_time + segment['start'],
                            "speaker": "mic",
                            "text": segment['text'].strip()
                        })
                elif source_name == 'sys':
                    self.sys_transcript_with_timestamps.extend(result['segments'])
                    self.sys_audio_for_diarization.append(audio_buffer.copy())

        print(f"[{source_name.upper()}] Processor thread finished.")
        if all(not t.is_alive() for t in self.processor_threads if t is not threading.current_thread()):
             self.gui_queue.put({"type": "processors_finished"})

    def run_diarization(self):
        temp_wav_path = f"temp_diarization_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        try:
            full_audio = np.concatenate(self.sys_audio_for_diarization)
            
            with wave.open(temp_wav_path, 'w') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(WHISPER_SAMPLE_RATE)
                wf.writeframes((full_audio * 32767).astype(np.int16).tobytes())

            num_speakers_str = self.num_speakers_entry.get()
            num_speakers = int(num_speakers_str) if num_speakers_str.isdigit() and int(num_speakers_str) > 0 else None
            
            print(f"[INFO] Running diarization for {num_speakers or 'an unknown number of'} speakers...")
            diarization = self.diarization_pipeline(temp_wav_path, num_speakers=num_speakers)

            all_words = [word for segment in self.sys_transcript_with_timestamps for word in segment.get('words', [])]
            if not all_words: 
                print("[INFO] No words with timestamps found for diarization."); return

            diarized_phrases = []
            current_phrase = None
            for word_info in all_words:
                word_text = word_info['word']
                mid_time = (word_info['start'] + word_info['end']) / 2.0
                
                speaker_label = "UNKNOWN"
                for turn, _, label in diarization.itertracks(yield_label=True):
                    if turn.start <= mid_time <= turn.end: speaker_label = label; break
                
                if current_phrase and current_phrase["speaker"] == speaker_label:
                    current_phrase["text"] += " " + word_text
                else:
                    if current_phrase: diarized_phrases.append(current_phrase)
                    current_phrase = {"start": word_info['start'], "speaker": speaker_label, "text": word_text}
            if current_phrase: diarized_phrases.append(current_phrase)
            
            for phrase in diarized_phrases:
                self.combined_transcript.append({
                    "start": self.recording_start_time + phrase['start'],
                    "speaker": phrase['speaker'],
                    "text": phrase['text'].strip()
                })

            self.sys_textbox.configure(state="normal"); self.sys_textbox.delete("1.0", "end")
            
            unique_speakers = sorted(list({p['speaker'] for p in diarized_phrases}))
            speaker_to_idx_map = {name: i for i, name in enumerate(unique_speakers)}

            for i, color in enumerate(self.speaker_colors):
                self.sys_textbox.tag_config(f"speaker_{i}", foreground=color)

            for phrase in diarized_phrases:
                speaker_idx = speaker_to_idx_map.get(phrase['speaker'], 0) % len(self.speaker_colors)
                display_text = f"[{phrase['speaker']}] {phrase['text']}"
                self.gui_queue.put({"type": "diarization_update", "text": display_text, "speaker_idx": speaker_idx})
            
            print("[INFO] Diarization complete.")

        except Exception as e:
            print(f"[ERROR] Diarization failed: {e}"); traceback.print_exc()
            self.gui_queue.put({"type": "status_update", "text": "Diarization failed.", "color": "red"})
        finally:
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            self.gui_queue.put({"type": "diarization_finished"})


    def generate_summary_threaded(self):
        self.summary_button.configure(state="disabled", text="Generating...")
        self.summary_textbox.configure(state="normal"); self.summary_textbox.delete("1.0", "end"); self.summary_textbox.configure(state="disabled")
        threading.Thread(target=self.run_summary_generation, daemon=True).start()

    def run_summary_generation(self):
        if not self.combined_transcript:
            self.gui_queue.put({"type": "summary_result", "data": "Transcript is empty."})
            return

        sorted_transcript = sorted(self.combined_transcript, key=lambda x: x['start'])
        full_transcript = "\n".join([f"{item['speaker']}: {item['text']}" for item in sorted_transcript])

        if not full_transcript.strip():
            self.gui_queue.put({"type": "summary_result", "data": "Transcript is empty."})
            return
        
        api_key = self.api_key_entry.get().strip()
        self.update_status("Attempting local summary (Ollama)...", "blue")
        summary, error = summarize_with_ollama(full_transcript)
        if error and "Not implemented" not in error:
            if api_key:
                self.update_status("Ollama failed. Trying OpenAI...", "orange")
                summary, error = summarize_with_openai(full_transcript, api_key)
            else:
                summary = f"Ollama failed ({error}). Provide an OpenAI API Key for fallback."
        if error and summary is None: summary = f"Summarization failed: {error}"
        self.update_status("Summary complete.", "green")
        self.gui_queue.put({"type": "summary_result", "data": summary})

    def export_transcript(self):
        if not self.combined_transcript:
            self.update_status("Nothing to export.", "orange"); return

        # Sort the combined transcript by the start timestamp
        sorted_transcript = sorted(self.combined_transcript, key=lambda x: x['start'])

        # Create a mapping for speaker labels to be more user-friendly
        unique_sys_speakers = sorted(list({item['speaker'] for item in sorted_transcript if item['speaker'] != 'mic'}))
        speaker_map = {label: f"speaker {i+1}" for i, label in enumerate(unique_sys_speakers)}
        speaker_map['mic'] = 'mic'

        # Format the transcript into the desired sequential flow
        transcript_flow = []
        for item in sorted_transcript:
            label = speaker_map.get(item['speaker'], item['speaker'])
            transcript_flow.append(f"{label}: {item['text']}")

        final_transcript_string = "\n".join(transcript_flow)

        # Save to a .txt file
        filepath = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(final_transcript_string)
            print(f"[INFO] Transcript exported to {os.path.abspath(filepath)}")
            self.update_status(f"Exported to {os.path.basename(filepath)}", "green")
        except Exception as e:
            print(f"[ERROR] Export failed: {e}"); self.update_status("Export failed.", "red")

if __name__ == "__main__":
    app = App()
    app.after(100, app.start_app_logic)
    app.mainloop()