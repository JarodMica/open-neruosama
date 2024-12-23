import base64
import logging
import io
import json
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import queue
import threading

from dotenv import load_dotenv
from typing import BinaryIO
from numpy import ndarray
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

load_dotenv()
open_ai_project_id = os.getenv('OPENAI_PROJECT_ID')
open_ai_api_key = os.getenv('OPENAI_API_KEY')

import whisper

class ASR:
    # The openai-whisper package is used because there's an issue with faster_whisper/whisperx
    # that causes the program to execute too slowly for GPT-SoVITS streaming
    # Found this out by using SpeechRecogntion, but don't want to use it so just opting for whisper
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(self):
        p = self.parameters
        model_name = p.get("asr_name")
        assert model_name, "No asr_name passed, please configure it."
        if model_name == "whisper":
            try:
                p_w = p.get("whisper")
                whisper_model_size = p_w.get("model_size", "base")
                compute_type = p_w.get("compute_type")
                download_root = p_w.get("download_root")
            except Exception as e:
                self.logger.error(f"ASR config not set up correctly: {e}")
            self.model = whisper.load_model(
                whisper_model_size, 
                device="cuda", 
                # compute_type=compute_type, 
                download_root=download_root
            )
        else:
            self.logger.error("PANIC")
            self.logger.error("asr_name not specified, please do it in the conf file.")
    
    def run(self, audio: str | ndarray) -> str:
        m = self.model
        # segments, info = m.transcribe(audio, beam_size=5)
        segments = m.transcribe(audio)
        # results = list(segments)
        # text = results[0].text
        return segments["text"]

import random
from llama_cpp import Llama
from openai import OpenAI

class LLM:
    def __init__(self, parameters):
        self.parameters = parameters
        self.local_parameters = parameters.get("local", None)
        self.api_parameters = parameters.get("api", None)
        self.model = None
        self.local = None
        
    def load_model(self):
        self.local = self.parameters.get("llm_local", True)
        if self.local:
            try:
                p_local = self.local_parameters
                model_path = p_local.get("llm_path")
                n_gpu_layers = int(p_local.get("n_gpu_layers"))
                seed = int(p_local.get("seed"))
                n_ctx = int(p_local.get("n_ctx"))
                chat_format = p_local.get("chat_format")
            except Exception as e:
                raise(f"Local LLM config file issue: {e}")
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                n_ctx=n_ctx,
                chat_format=chat_format,
                verbose=False
            )
        else:
            global open_ai_project_id, open_ai_api_key
            p_api = self.api_parameters
            if not open_ai_project_id:
                open_ai_project_id = p_api.get("project_id")
                open_ai_api_key = p_api.get("api_key")
            self.model = OpenAI(
                project=open_ai_project_id,
                api_key=open_ai_api_key
            )
        
    def generate_chat_completion_openai(self, messages_dict):
        stream = self.model.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_dict,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def generate_chat_completion_openai_v1_stream(self, messages_dict):
        p_l = self.local_parameters
        temperature = float(p_l.get("temperature"))
        top_p = float(p_l.get("top_p"))
        top_k = int(p_l.get("top_k"))
        max_tokens = int(p_l.get("max_tokens"))
        stream = self.model.create_chat_completion_openai_v1(
            messages=messages_dict,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    def run(self, messages: list):
        assert self.model, "Model is not loaded, use load_model() first"
        assert type(messages) == list, "Messages needs to be a list of dicts"
        if self.local:
            yield from self.generate_chat_completion_openai_v1_stream(messages)
        else:
            yield from self.generate_chat_completion_openai(messages)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITSPipeline
from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config as GPTSoVITSConfig

class TTS:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.loaded = False

    def load_model(self):
        tts_name = self.parameters.get("tts_name")
        if not tts_name:
            raise ValueError("No tts_name provided in the config.")
        
        # For example, if tts_name == "gpt-sovits", load GPT-SoVITS
        if tts_name == "gpt-sovits":
            # try:
            p_tts = self.parameters.get("gpt-sovits", {})
            # Example keys you'd store in your config
            config_path = p_tts.get("config_path")
            gpt_model_path = p_tts.get("gpt_model_path")
            sovits_model_path = p_tts.get("sovits_model_path")

            # Example initialization (uncomment if actually using GPT-SoVITS)
            self.tts_config = GPTSoVITSConfig(config_path)
            self.model = GPTSoVITSPipeline(self.tts_config)
            # self.model.init_t2s_weights(gpt_model_path)
            # self.model.init_vits_weights(sovits_model_path)

            self.loaded = True
            self.logger.info("GPT-SoVITS TTS model loaded successfully.")
            # except Exception as e:
            #     self.logger.error(f"Failed to load GPT-SoVITS TTS model: {e}")
            #     raise e
        else:
            # Extend logic for other TTS backends
            error_msg = f"Unsupported TTS model: {tts_name}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
    def run(self, tts_inputs: dict):
        if not self.loaded or not self.model:
            raise RuntimeError("TTS model is not loaded. Call load_model() first.")

        # Retrieve defaults from config for GPT-SoVITS
        p_tts = self.parameters.get("gpt-sovits", {})
        merged_inputs = {
            "text": tts_inputs.get("text", ""),  # text is mandatory
            "text_lang": tts_inputs.get("text_lang", p_tts.get("default_text_lang", "en")),
            "ref_audio_path": tts_inputs.get("ref_audio_path", None),
            "prompt_text": tts_inputs.get("prompt_text", ""),
            "prompt_lang": tts_inputs.get("prompt_lang", p_tts.get("default_prompt_lang", "en")),
            "top_k": tts_inputs.get("top_k", p_tts.get("top_k", 5)),
            "top_p": tts_inputs.get("top_p", p_tts.get("top_p", 1.0)),
            "temperature": tts_inputs.get("temperature", p_tts.get("temperature", 1.0)),
            "text_split_method": tts_inputs.get("text_split_method", p_tts.get("text_split_method", "cut0")),
            "batch_size": tts_inputs.get("batch_size", p_tts.get("batch_size", 1)),
            "batch_threshold": tts_inputs.get("batch_threshold", p_tts.get("batch_threshold", 0.75)),
            "split_bucket": tts_inputs.get("split_bucket", p_tts.get("split_bucket", True)),
            "speed_factor": tts_inputs.get("speed_factor", p_tts.get("speed_factor", 1.0)),
            "fragment_interval": tts_inputs.get("fragment_interval", p_tts.get("fragment_interval", 0.3)),
            "seed": tts_inputs.get("seed", p_tts.get("seed", 1234)),
            "return_fragment": tts_inputs.get("return_fragment", p_tts.get("return_fragment", True)),
            "parallel_infer": tts_inputs.get("parallel_infer", p_tts.get("parallel_infer", False)),
            "repetition_penalty": tts_inputs.get("repetition_penalty", p_tts.get("repetition_penalty", 1.2)),
        }

        self.logger.debug(f"GPT-SoVITS final inference inputs: {merged_inputs}")
        yield from self.model.run_generator(merged_inputs)
        

import gc

class NeuroSama:
    def __init__(self, asr_params=None, llm_params=None, tts_params=None):
        self.asr_params = asr_params
        self.llm_params = llm_params
        self.tts_params = tts_params
        self.asr = None
        self.llm = None
        self.tts = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_all_models(self):
        self.load_asr()
        self.load_llm()
        self.load_tts()
        
    def _unload_model(self, model_name):
        model = getattr(self, model_name)
        if model:
            self.logger.info(f"Unloading existing {model_name.upper()} model...")
            delattr(self, model_name)
            setattr(self, model_name, None)
            gc.collect()
            self.logger.info(f"{model_name.upper()} model unloaded successfully.")
        else:
            self.logger.debug(f"No existing {model_name.upper()} model found")
        
    def load_asr(self):
        self._unload_model("asr")
        p = self.asr_params
        self.logger.info("Loading ASR model")
        try:
            asr = ASR(p)
            asr.load_model()
            self.asr = asr
            self.logger.info("ASR model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {e}")
        
    def load_llm(self):
        self._unload_model("llm")
        p = self.llm_params
        self.logger.info("Loading LLM model")
        try:
            llm = LLM(p)
            llm.load_model()
            self.llm = llm
            self.logger.info("LLM model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
        
    def load_tts(self):
        self._unload_model("tts")
        p = self.tts_params
        self.logger.info("Loading TTS model")
        # try:
        tts = TTS(p)
        tts.load_model()
        self.tts = tts
        self.logger.info("TTS model loaded successfully.")
        # except Exception as e:
        #     self.logger.error(f"Failed to load TTS model: {e}")
    
    def generate_llm_output(self, messages: list):
        response = self.llm.run(messages)
        for chunk in response:
            print(chunk, end="")

def get_configuration(path: str, config_key: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parameters = data[f"{config_key}"]
    return parameters

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def audio_playback_thread(audio_queue: queue.Queue, sample_rate: int):
    """
    A background thread that plays audio fragments as they become available
    in the queue. You can adapt this logic if you have a different audio
    playback method.
    """
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    stream = sd.OutputStream(dtype='float32')
    stream.start()
    try:
        while True:
            audio_fragment = audio_queue.get()
            try:
                if audio_fragment is None:
                    # Sentinel received, end thread
                    break
                stream.write(audio_fragment)
            finally:
                audio_queue.task_done()
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    asr_key = "1"
    llm_key = "2"
    tts_key = "1"
    base64_image = image_to_base64("test_image_1.png")

    messages = [{"role": "system", "content": "You are a nice, helpful AI voice assistant that is named Vivy.  You are to chat with the user!"}]

    asr_params = get_configuration("conf_asr.json", asr_key)
    llm_params = get_configuration("conf_llm.json", llm_key)
    tts_params = get_configuration("conf_tts.json", tts_key)

    logger = logging.getLogger()
    
    neuro = NeuroSama(asr_params=asr_params, llm_params=llm_params, tts_params=tts_params)
    neuro.load_asr()
    neuro.load_llm()
    neuro.load_tts()
    
    tts_sample_rate = 32000
    if hasattr(neuro.tts.model, "configs"):
        tts_sample_rate = getattr(neuro.tts.model.configs, "sampling_rate", 32000)

    r = sr.Recognizer()
    r.pause_threshold = 1.5
    r.phrase_threshold = 1.0
    mic = sr.Microphone()
    
    audio_queue = queue.Queue(maxsize=100)
    playback_thread = threading.Thread(
        target=audio_playback_thread,
        args=(audio_queue, tts_sample_rate),
        daemon=True
    )
    playback_thread.start()

    try:
        while True:
            print("\nPlease speak your message (say 'quit' to exit).")
            # while True:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=1.0)
                print("Listening...")
                audio_data = r.listen(source, timeout=None, phrase_time_limit=60)

            # Convert microphone data to WAV and pass to your ASR
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            wav_stream = io.BytesIO(wav_bytes)
            audio_array, sampling_rate = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)
            spoken_text = neuro.asr.run(audio_array).strip()
            
            logger.info(f"User said: {spoken_text}")

            if spoken_text.lower() == "quit":
                print("User requested to quit. Exiting.")
                break

            if not spoken_text:
                print("No speech detected. Please try again.")
                continue

            # Append user input to conversation
            messages.append({"role": "user", "content": spoken_text})

            # We'll accumulate partial tokens in a 'partial_buffer'
            # and watch for punctuation to split sentences
            partial_buffer = ""
            char_count = 0
            waiting_for_punctuation = False
            assistant_buffer = ""

            print("Assistant: ", end="", flush=True)
            for token in neuro.llm.run(messages):
                print(token, end="", flush=True)
                partial_buffer += token
                assistant_buffer += token
                char_count += len(token)

                # Once we've printed ~100 characters, start waiting for punctuation
                if not waiting_for_punctuation and char_count >= 100:
                    waiting_for_punctuation = True

                if waiting_for_punctuation:
                    # If we see punctuation, let's treat that as a sentence boundary
                    if any(punct in token for punct in [".", "!", "?"]):
                        tts_inputs = {
                            "text": partial_buffer,
                            "ref_audio_path" : "example_41.wav",
                            "prompt_text" : "I'm very new to this But I want to learn So we got another leg I probably should have tried to fit it Oh no"
                        }
                        synthesis_result = neuro.tts.run(tts_inputs)
                        for sr_out, audio_fragment in synthesis_result:
                            audio_queue.put(audio_fragment)

                        # Optionally enqueue a short silence
                        silence_duration = 0.5
                        silence_samples = int(sr_out * silence_duration)
                        silence = np.zeros(silence_samples, dtype='float32')
                        audio_queue.put(silence)

                        # Reset partial buffer
                        partial_buffer = ""
                        char_count = 0
                        waiting_for_punctuation = False

            # Once the LLM is done generating, we have leftover text in `partial_buffer`.
            if partial_buffer.strip():
                tts_inputs = {
                    "text": partial_buffer,
                }
                synthesis_result = neuro.tts.run(tts_inputs)
                for sr_out, audio_fragment in synthesis_result:
                    audio_queue.put(audio_fragment)

                # Insert a bit of silence
                silence_duration = 0.5
                silence_samples = int(sr_out * silence_duration)
                silence = np.zeros(silence_samples, dtype='float32')
                audio_queue.put(silence)

            # Add the entire assistant response to conversation
            messages.append({"role": "assistant", "content": assistant_buffer})

    finally:
        # Clean shutdown: send sentinel to audio thread and wait
        audio_queue.put(None)
        audio_queue.join()
        playback_thread.join()
        print("\nAll done. Goodbye!")
