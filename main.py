import customtkinter as ctk
import threading
import queue
import time
import json
import os
import traceback
import subprocess
from datetime import datetime

from summarizer import summarize_with_ollama, summarize_with_openai  # Ensure this file exists with the required functions

# --- Backend Imports ---
import numpy as np
import whisper
import torch
import sounddevice as sd
import resampy


# --- App Configuration & Globals ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
WHISPER_SAMPLE_RATE = 16000
IS_GPU_AVAILABLE = torch.cuda.is_available()
print(f"[INFO] CUDA available: {IS_GPU_AVAILABLE}")


# --- Main Application Class ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local Real-Time Transcriber")
        self.geometry("1100x800")

        self.is_recording = False
        self.app_running = True
        self.gui_queue = queue.Queue()
        self.mic_data_queue = queue.Queue()
        self.sys_data_queue = queue.Queue()
        self.capture_threads = {"mic": {"thread": None, "stop_event": threading.Event()}, "sys": {"thread": None, "stop_event": threading.Event()}}
        self.processor_threads = []
        self.transcript_data = {"mic": [], "sys": []}
        self.last_interim_text = {"mic": "", "sys": ""}
        self.whisper_model = None
        # FIX 1: Added a variable to store the name of the currently loaded model.
        self.current_model_name = ""
        self.whisper_lock = threading.Lock()  # <-- Add this line

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_ui()

    def start_app_logic(self):
        self.after(50, self.process_gui_queue)
        self.populate_device_menus()
        self.load_whisper_model()
        self.start_capture_threads()

    def setup_ui(self):
        self.grid_columnconfigure((0, 1), weight=1); self.grid_rowconfigure(2, weight=1)
        self.top_frame = ctk.CTkFrame(self, corner_radius=10); self.top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.top_frame.grid_columnconfigure(2, weight=1)
        self.toggle_button = ctk.CTkButton(self.top_frame, text="Start Recording", command=self.toggle_recording, width=150); self.toggle_button.grid(row=0, column=0, padx=10, pady=10)
        self.language_menu = ctk.CTkOptionMenu(self.top_frame, values=["english", "italian", "german", "spanish", "french"], command=self.on_language_change); self.language_menu.set("english"); self.language_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.status_label = ctk.CTkLabel(self.top_frame, text="Status: Initializing...", text_color="gray"); self.status_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.device_frame = ctk.CTkFrame(self, corner_radius=10); self.device_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.device_frame.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(self.device_frame, text="Microphone:").grid(row=0, column=0, padx=(10,0))
        self.mic_device_menu = ctk.CTkOptionMenu(self.device_frame, values=["loading..."], command=self.on_mic_device_change); self.mic_device_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.device_frame, text="System Audio (Monitor):").grid(row=0, column=2, padx=(10,0))
        self.sys_device_menu = ctk.CTkOptionMenu(self.device_frame, values=["loading..."], command=self.on_sys_device_change); self.sys_device_menu.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
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
            native_rate = 16000  # Force to 16kHz
            chunk_size = 1024
            print(f"[{source_name.upper()}] Opening sounddevice stream on '{dev_info['name']}' at {native_rate}Hz using blocking read...")

            with sd.InputStream(device=device_index,
                                channels=1,
                                samplerate=native_rate,
                                dtype='float32',
                                blocksize=chunk_size) as stream:

                print(f"[{source_name.upper()}] Stream opened successfully.")
                while not stop_event.is_set():
                    indata, overflowed = stream.read(chunk_size)
                    # print(f"[{source_name.upper()}] Read {indata.shape} samples, overflowed={overflowed}")
                    if overflowed:
                        print(f"[WARN] Mic stream overflowed!")

                    self.gui_queue.put({"type": "volume_update", "source": "mic", "volume": np.linalg.norm(indata) * 10})

                    if self.is_recording:
                        try:
                            data_queue.put(indata)
                        except Exception as e:
                            print(f"[ERROR] Mic resampling failed in read loop: {e}")

        except Exception as e:
            print(f"[ERROR] Mic capture failed: {e}")
            traceback.print_exc()
        print(f"[{source_name.upper()}] Capture thread stopped.")


    def parec_capture_thread(self, device_name, data_queue, stop_event):
        source_name = "sys"; parec = None; PAREC_RATE = 44100; CHUNK = 2048
        try:
            command = ["parec", f"--device={device_name}", "--format=s16le", f"--rate={PAREC_RATE}", "--channels=1"]
            parec = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[INFO] System audio stream opened for '{device_name}'")
            while not stop_event.is_set():
                raw = parec.stdout.read(CHUNK)
                if not raw:
                    time.sleep(0.01)
                    continue
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
        if self.is_recording: self.stop_recording()
        else: self.start_recording()

    def start_recording(self):
        if self.whisper_model is None:
            self.update_status("Error: Whisper model not loaded.", "red")
            return
        self.transcript_data = {"mic": [], "sys": []}
        self.last_interim_text = {"mic": "", "sys": ""}
        for textbox in [self.mic_textbox, self.sys_textbox]:
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.configure(state="disabled")
        self.toggle_button.configure(text="Stop Recording", fg_color="#DB4437", hover_color="#C53727")
        for widget in [self.summary_button, self.export_button, self.language_menu, self.mic_device_menu, self.sys_device_menu]:
            widget.configure(state="disabled")
        
        # Set is_recording to True BEFORE starting processor threads
        self.is_recording = True

        mic_processor = threading.Thread(target=self.audio_processor_thread, args=("mic", self.mic_data_queue), daemon=True)
        sys_processor = threading.Thread(target=self.audio_processor_thread, args=("sys", self.sys_data_queue), daemon=True)
        self.processor_threads = [mic_processor, sys_processor]
        mic_processor.start()
        sys_processor.start()
        
        self.update_status("Status: Recording...", "green")

    def stop_recording(self):
        self.is_recording = False
        self.update_status("Status: Finalizing...", "orange")

    def on_recording_finished(self):
        self.update_status("Status: Idle", "gray")
        self.toggle_button.configure(text="Start Recording", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        for widget in [self.summary_button, self.export_button, self.language_menu, self.mic_device_menu, self.sys_device_menu]:
            if widget not in (self.mic_device_menu, self.sys_device_menu) or "No" not in widget.get():
                widget.configure(state="normal")
        print("[INFO] Recording session finished.")

    def on_language_change(self, choice):
        self.load_whisper_model()

    def load_whisper_model(self):
        lang = self.language_menu.get()
        model_name = "base.en" if lang == "english" else "base"
        
        # FIX 1: Check against our stored model name variable instead of a non-existent attribute.
        if self.whisper_model and self.current_model_name == model_name:
            print(f"[INFO] Model '{model_name}' is already loaded.")
            return

        self.update_status(f"Loading {model_name} model...", "orange")
        def load():
            try:
                device = "cuda" if IS_GPU_AVAILABLE else "cpu"
                self.whisper_model = whisper.load_model(model_name, device=device)
                # FIX 1: Store the name of the successfully loaded model.
                self.current_model_name = model_name
                self.update_status("Status: Ready", "gray")
                print(f"[INFO] Model '{model_name}' loaded on {device}.")
            except Exception as e:
                self.update_status("Error loading model!", "red")
                print(f"[ERROR] Model load fail: {e}")
                self.whisper_model = None
                self.current_model_name = "" # Reset on failure
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
                        self.transcript_data[source].append(text)
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
        except queue.Empty: pass
        finally: self.after(50, self.process_gui_queue)

    def audio_processor_thread(self, source_name, data_queue):
        INTERIM_TIMEOUT = 0.7; PHRASE_TIMEOUT = 2.5
        audio_buffer = np.array([], dtype=np.float32)
        last_audio_time = last_interim_time = time.time()
        
        print(f"[{source_name.upper()}] Processor thread started.")
        
        # Phase 1: While recording, process incoming data
        while self.is_recording:
            try:
                now = time.time()
                try:
                    # Wait for data with timeout to avoid busy loop
                    chunk = data_queue.get(timeout=0.5)
                    audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                    last_audio_time = now
                    # Drain any additional chunks quickly
                    while True:
                        chunk = data_queue.get_nowait()
                        audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                        last_audio_time = now
                except queue.Empty:
                    pass

                time_since_last_audio = now - last_audio_time
                time_since_interim = now - last_interim_time
                
                is_final = time_since_last_audio > PHRASE_TIMEOUT
                is_interim = time_since_interim > INTERIM_TIMEOUT
                force_process = False

                if (is_interim or is_final or force_process) and audio_buffer.size > 1000:
                    lang = self.language_menu.get() if self.language_menu.get() != "english" else None
                    with self.whisper_lock:  # <-- Add this block
                        text = self.whisper_model.transcribe(audio_buffer, language=lang, fp16=IS_GPU_AVAILABLE)['text'].strip()
                    if text:
                        self.gui_queue.put({"type": "transcript_update", "source": source_name, "text": text, "is_final": is_final or force_process})
                    last_interim_time = now

                if is_final or force_process:
                    audio_buffer = np.array([], dtype=np.float32)
                    self.last_interim_text[source_name] = ""

                time.sleep(0.1)
            except Exception as e: print(f"[{source_name.upper()}] Processor error: {e}"); traceback.print_exc(); break

        # Phase 2: After recording stops, drain any remaining data in the queue
        while not data_queue.empty():
            try:
                chunk = data_queue.get_nowait()
                audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
            except queue.Empty:
                break
        # If there's any audio left, process it as final
        if audio_buffer.size > 1000:
            lang = self.language_menu.get() if self.language_menu.get() != "english" else None
            with self.whisper_lock:  # <-- Add this block
                text = self.whisper_model.transcribe(audio_buffer, language=lang, fp16=IS_GPU_AVAILABLE)['text'].strip()
            if text:
                self.gui_queue.put({"type": "transcript_update", "source": source_name, "text": text, "is_final": True})
            self.last_interim_text[source_name] = ""

        print(f"[{source_name.upper()}] Processor thread finished.")
        # Check if this is the last active processor thread
        if all(not t.is_alive() for t in self.processor_threads if t is not threading.current_thread()):
             self.gui_queue.put({"type": "processors_finished"})


    def generate_summary_threaded(self):
        self.summary_button.configure(state="disabled", text="Generating...")
        self.summary_textbox.configure(state="normal"); self.summary_textbox.delete("1.0", "end"); self.summary_textbox.configure(state="disabled")
        threading.Thread(target=self.run_summary_generation, daemon=True).start()

    def run_summary_generation(self):
        mic_text = "\n".join(self.transcript_data["mic"]).strip(); sys_text = "\n".join(self.transcript_data["sys"]).strip()
        full_transcript = f"--- USER/MICROPHONE TRANSCRIPT ---\n{mic_text}\n\n--- CALL/SYSTEM AUDIO TRANSCRIPT ---\n{sys_text}"
        if not (mic_text or sys_text):
            self.gui_queue.put({"type": "summary_result", "data": "Transcript is empty."})
            return
        api_key = self.api_key_entry.get().strip()
        self.update_status("Attempting local summary (Ollama)...", "blue")
        summary, error = summarize_with_ollama(full_transcript)
        if error and "Not implemented" not in error: # Simple check for placeholder
            if api_key:
                self.update_status("Ollama failed. Trying OpenAI...", "orange")
                summary, error = summarize_with_openai(full_transcript, api_key)
            else:
                summary = f"Ollama failed ({error}). Provide an OpenAI API Key for fallback."
        if error and summary is None:
            summary = f"Summarization failed: {error}"
        self.update_status("Summary complete.", "green")
        self.gui_queue.put({"type": "summary_result", "data": summary})

    def export_transcript(self):
        export_data = {"metadata": {"export_time": datetime.now().isoformat(), "language": self.language_menu.get()}, "microphone_transcript": self.transcript_data["mic"], "system_audio_transcript": self.transcript_data["sys"]}
        if not export_data["microphone_transcript"] and not export_data["system_audio_transcript"]:
            self.update_status("Nothing to export.", "orange")
            return
        filepath = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filepath, "w", encoding='utf-8') as f: json.dump(export_data, f, indent=4)
            print(f"[INFO] Transcript exported to {os.path.abspath(filepath)}")
            self.update_status(f"Exported to {os.path.basename(filepath)}", "green")
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            self.update_status("Export failed.", "red")

if __name__ == "__main__":
    app = App()
    app.after(100, app.start_app_logic)
    app.mainloop()