import math
import json
import librosa
import torch
import torchaudio
import accelerate
import safetensors
import numpy as np
import os
import yaml
from IPython.display import display, Audio

import torchvision
import random
import numpy as np
import whisper
from librosa.feature import chroma_stft
from librosa.effects import pitch_shift

from models.codec.coco.rep_coco_model import CocoContentStyle, CocoContent, CocoStyle
from models.svc.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.codec.melvqgan.melspec import MelSpectrogram
from models.codec.amphion_codec.vocos import Vocos

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.util import load_config
from models.svc.vevo2.qwen_utils import gen_chat_prompt
from evaluation.metrics.f0.f0_corr import extract_f0_hz

from transformers.utils import is_flash_attn_2_available

supported_flash_attn = False
if not torch.cuda.is_available():
    print("No CUDA available")
    supported_flash_attn = False

# To check if flash attention is supported
if is_flash_attn_2_available():
    supported_flash_attn = True
    print("Flash Attention is supported")
else:
    print("Flash Attention is not supported")


# Coco Tokenizer
def build_coco_model(coco_cfg, device, loading_decoder=False):
    coco_model_type = getattr(coco_cfg, "coco_type", "content_style")
    if coco_model_type == "content_style":
        model = CocoContentStyle(
            cfg=coco_cfg, construct_only_for_quantizer=not loading_decoder
        )
    elif coco_model_type == "content":
        model = CocoContent(
            cfg=coco_cfg, construct_only_for_quantizer=not loading_decoder
        )
    elif coco_model_type == "style":
        model = CocoStyle(
            cfg=coco_cfg, construct_only_for_quantizer=not loading_decoder
        )
    else:
        raise ValueError(f"Unknown coco type: {coco_model_type}")

    model.eval()
    model.to(device)
    return model


# Flow Matching Transformer
def build_fmt_model(cfg, device):
    model = FlowMatchingTransformer(cfg=cfg.model.flow_matching_transformer)
    model.eval()
    model.to(device)
    return model


# Autoregressive Transformer
def build_ar_model(ckpt_path, device):
    model_kwargs = {
        "device_map": device,
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }

    # Only add flash attention parameter if supported
    if supported_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg.model.pretrained_model_path, **model_kwargs
    # )
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, **model_kwargs)

    model.eval()
    return model


# Melspectrogram Extractor
def build_mel_model(cfg, device):
    mel_model = MelSpectrogram(
        sampling_rate=cfg.preprocess.sample_rate,
        n_fft=cfg.preprocess.n_fft,
        num_mels=cfg.preprocess.num_mels,
        hop_size=cfg.preprocess.hop_size,
        win_size=cfg.preprocess.win_size,
        fmin=cfg.preprocess.fmin,
        fmax=cfg.preprocess.fmax,
    )
    mel_model.eval()
    mel_model.to(device)
    return mel_model


# Vocoder
def build_vocoder_model(cfg, device):
    vocoder_model = Vocos(cfg=cfg.model.vocos)
    vocoder_model.eval()
    vocoder_model.to(device)
    return vocoder_model


def load_checkpoint(build_model_func, cfg, ckpt_path, device):
    model = build_model_func(cfg, device)
    accelerate.load_checkpoint_and_dispatch(model, ckpt_path)
    return model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1e6:
        return f"{total_params} params"  # Parameters
    elif total_params < 1e9:
        return f"{total_params / 1e6:.2f} M"  # Millions
    else:
        return f"{total_params / 1e9:.2f} B"  # Billions


def load_wav(wav_path, device, used_duration=None):
    if wav_path is None:
        speech = np.zeros((0))  # [T]
        speech_tensor = torch.zeros(1, 0).to(device)  # [1, T]
        speech16k = torch.zeros(1, 0).to(device)  # [1, T']
    else:
        speech = librosa.load(wav_path, sr=24000)[0]  # [T]

        if used_duration is not None:
            speech = speech[: int(used_duration * 24000)]

        speech_tensor = torch.tensor(speech).unsqueeze(0).to(device)  # [1, T]
        speech16k = torchaudio.functional.resample(
            speech_tensor, 24000, 16000
        )  # [1, T']

    return speech, speech_tensor, speech16k


def display_audio_in_notebook(wav, rate=24000):
    display(Audio(wav, rate=rate))


def save_audio(
    waveform, sr=24000, output_path=None, target_sample_rate=None, target_db=-25.0
):
    """
    waveform: [1, T]
    """
    if target_sample_rate is not None and sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
    else:
        target_sample_rate = sr

    rms = torch.sqrt(torch.mean(waveform**2))
    current_db = 20 * torch.log10(rms + 1e-9)

    gain = target_db - current_db
    normalized_waveform = waveform * (10 ** (gain / 20))

    torchaudio.save(output_path, normalized_waveform, target_sample_rate)
    return output_path


def extract_special_ids(text, prefix="content_style"):
    """
    Extract all audio IDs from a string containing <|content_style_X|> or <|prosody_X|> tags

    Args:
        text (str): A string containing audio tags
        prefix (str): The prefix of the audio tags

    Returns:
        list: A list of all audio IDs
    """
    import re

    if prefix == "content_style":
        # Use regex to match all <|content_style_X|> format tags
        pattern = r"<\|content_style_(\d+)\|>"
    elif prefix == "prosody":
        pattern = r"<\|prosody_(\d+)\|>"
    else:
        raise ValueError(f"Unknown prefix: {prefix}")

    # Find all matches and extract the numeric part
    audio_ids = re.findall(pattern, text)

    # Convert string IDs to integers
    audio_ids = [int(id) for id in audio_ids]

    return audio_ids


class Vevo2InferencePipeline:
    def __init__(
        self,
        prosody_tokenizer_ckpt_path=None,
        content_style_tokenizer_ckpt_path=None,
        ar_cfg_path=None,
        ar_ckpt_path=None,
        fmt_cfg_path=None,
        fmt_ckpt_path=None,
        vocoder_cfg_path=None,
        vocoder_ckpt_path=None,
        device=None,
        use_vllm=False,
    ):
        self.device = device
        self.use_vllm = use_vllm

        self.prosody_tokenizer_ckpt_path = prosody_tokenizer_ckpt_path
        self.content_style_tokenizer_ckpt_path = content_style_tokenizer_ckpt_path

        if ar_cfg_path is not None and ar_ckpt_path is not None:
            self.ar_cfg = load_config(ar_cfg_path)

            assert not use_vllm, "VLLM is not supported yet"

            if use_vllm:
                pass
            else:
                # self.ar_model = load_checkpoint(
                #     build_ar_model, self.ar_cfg, ar_ckpt_path, device
                # )

                self.ar_model = build_ar_model(ar_ckpt_path, device)
                print(f"#Params of AR model: {count_parameters(self.ar_model)}")

            # self.ar_tokenizer = AutoTokenizer.from_pretrained(
            #     self.ar_cfg.preprocess.tokenizer_path, local_files_only=True
            # )
            self.ar_tokenizer = AutoTokenizer.from_pretrained(
                ar_ckpt_path, local_files_only=True
            )
        else:
            self.ar_cfg = None
            self.ar_model = None

        if fmt_cfg_path is not None and fmt_ckpt_path is not None:
            self.fmt_cfg = load_config(fmt_cfg_path)
            self.fmt_model = load_checkpoint(
                build_fmt_model, self.fmt_cfg, fmt_ckpt_path, device
            )
            print(f"#Params of Flow Matching model: {count_parameters(self.fmt_model)}")

            if getattr(
                self.fmt_cfg.model.flow_matching_transformer,
                "use_text_as_condition",
                False,
            ):
                self.fmt_use_text_as_condition = True
                self.fmt_text_tokenizer = AutoTokenizer.from_pretrained(
                    self.fmt_cfg.preprocess.tokenizer_path
                )
            else:
                self.fmt_use_text_as_condition = False

            self.init_coco_tokenizer()

        if vocoder_cfg_path is not None and vocoder_ckpt_path is not None:
            self.vocoder_cfg = load_config(vocoder_cfg_path)
            self.mel_model = build_mel_model(self.vocoder_cfg, device)
            self.vocoder_model = load_checkpoint(
                build_vocoder_model, self.vocoder_cfg, vocoder_ckpt_path, device
            )
            print(f"#Params of Vocoder model: {count_parameters(self.vocoder_model)}")

    def init_coco_tokenizer(self):
        ## Whisper ##
        self.whisper_model = whisper.load_model("medium", self.device)  # 1024 dim
        self.whisper_model.eval()

        self.use_normed_whisper = getattr(
            self.fmt_cfg.model.coco, "use_normed_whisper", False
        )
        if self.use_normed_whisper:
            whisper_stats = torch.load(
                self.fmt_cfg.model.coco.whisper_stats_path,
                map_location=self.device,
            )
            self.whisper_mean = whisper_stats["mean"]  # (1024,)
            self.whisper_std = whisper_stats["std"]  # (1024,)

        ## Prosody Tokenizer ##
        if self.ar_model is not None:
            self.style_tokenizer = load_checkpoint(
                build_coco_model,
                self.ar_cfg.model.coco_style,
                self.prosody_tokenizer_ckpt_path,
                self.device,
            )
            print(
                f"#Params of CocoStyle model: {count_parameters(self.style_tokenizer)}"
            )

        ## Content-Style Tokenizer ##
        self.content_style_tokenizer = load_checkpoint(
            build_coco_model,
            self.fmt_cfg.model.coco,
            self.content_style_tokenizer_ckpt_path,
            self.device,
        )
        print(
            f"#Params of CocoContentStyle model: {count_parameters(self.content_style_tokenizer)}"
        )

    @torch.no_grad()
    def extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.vocoder_cfg.preprocess.mel_mean) / math.sqrt(
            self.vocoder_cfg.preprocess.mel_var
        )
        return mel_feature

    def spec_augment(self, mel, height):
        """
        Args:
            mel: tensor (..., n_mels, frames)
            height: int 68-92 for default 80 mels
        """
        tgt = torchvision.transforms.functional.resize(mel, (height, mel.shape[-1]))
        if height >= mel.shape[-2]:
            return tgt[:, : mel.shape[-2], :]
        else:
            silence = tgt[:, -1:, :].repeat(1, mel.shape[-2] - height, 1)
            silence += torch.randn_like(silence) / 10
            return torch.cat((tgt, silence), 1)

    @torch.no_grad()
    def extract_whisper_features(self, wavs, frame_lens, spec_perturb=False):
        """
        Args:
            wavs: (B, T) at 16khz. Note that the max duration should be 30s
            frame_lens: (B,)
        Returns:
            features: (B, T, D)
        """
        # wavs: (batch, max_len)
        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.device)

        if spec_perturb:
            height = random.randint(68, 92)
            batch_mel = self.spec_augment(batch_mel, height)

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.whisper_model.embed_audio(batch_mel)

        max_len = int(frame_lens.max().item())
        mask = torch.arange(features.size(1), device=features.device).expand(
            len(frame_lens), -1
        ) < frame_lens.unsqueeze(1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))

        if features.shape[1] >= max_len:
            features = features[:, :max_len, :]
        else:
            padding_frames = max_len - features.shape[1]
            last_frame = features[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            features = torch.cat([features, padding], dim=1)

        if self.use_normed_whisper:
            features = (features - self.whisper_mean) / self.whisper_std

        return features

    @torch.no_grad()
    def extract_coco_codec(
        self,
        coco_codec_type,
        wav16k,
        wav24k_numpy,
        whisper_spec_perturb=False,
        frame_len_ratio=1.0,
        use_shifted_wav_to_extract_chromagram=False,
        use_shifted_wav_to_extract_whisper=False,
        pitch_shift_steps=0,
    ):
        """
        Args:
            coco_codec_type: "content", "style", or "content_style"
            wav16k: [1, T]
            wav24k_numpy: [T]
        Returns:
            codecs: [1, T]. Note that codecs might be not at 50Hz!
        """
        frame_len = len(wav24k_numpy) // self.fmt_cfg.preprocess.hop_size

        if use_shifted_wav_to_extract_chromagram:
            chromagram_feats = self.get_chromagram(
                pitch_shift(wav24k_numpy, sr=24000, n_steps=pitch_shift_steps),
                frame_len,
            )  # [T, 24]
        else:
            chromagram_feats = self.get_chromagram(wav24k_numpy, frame_len)  # [T, 24]

        chromagram_feats = (
            torch.tensor(chromagram_feats, dtype=torch.float)
            .unsqueeze(0)
            .to(self.device)
        )  # [1, T, 24]

        if frame_len_ratio != 1.0:
            raw_len = chromagram_feats.shape[1]
            # Convert [1, T, 24] to [1, 24, T] for interpolation on the last dimension
            chromagram_feats = chromagram_feats.transpose(1, 2)
            chromagram_feats = torch.nn.functional.interpolate(
                chromagram_feats,
                size=int(
                    raw_len * frame_len_ratio
                ),  # Explicitly specify the target length
                mode="linear",
                align_corners=False,
            )  # [1, 24, T']
            # Convert back to the original shape [1, T', 24]
            chromagram_feats = chromagram_feats.transpose(1, 2)
            print(
                f"Chromagram feats are sampled from {raw_len} to {chromagram_feats.shape[1]}, ratio = {frame_len_ratio}"
            )

        if use_shifted_wav_to_extract_whisper:
            wav16k = pitch_shift(
                wav16k.cpu().numpy()[0], sr=16000, n_steps=pitch_shift_steps
            )  # [T]
            wav16k = torch.tensor(wav16k).unsqueeze(0).to(self.device)  # [1, T]

        if coco_codec_type in ["content_style", "content"]:
            whisper_feats = self.extract_whisper_features(
                wav16k,
                torch.tensor([frame_len], dtype=torch.long).to(self.device),
                spec_perturb=whisper_spec_perturb,
            )  # [1, T, D]

        if coco_codec_type == "content_style":
            codecs, _ = self.content_style_tokenizer.quantize(
                whisper_feats.to(torch.float32), chromagram_feats.to(torch.float32)
            )
        elif coco_codec_type == "style":
            codecs, _ = self.style_tokenizer.quantize(
                chromagram_feats.to(torch.float32)
            )
        else:
            raise ValueError(f"Unknown coco type: {coco_codec_type}")

        return codecs

    def get_chromagram(self, speech, speech_frames):
        # [24, T] -> [T, 24]
        chromagram = chroma_stft(
            y=speech,
            sr=self.fmt_cfg.preprocess.sample_rate,
            n_fft=self.fmt_cfg.preprocess.n_fft,
            hop_length=self.fmt_cfg.preprocess.hop_size,
            win_length=self.fmt_cfg.preprocess.win_size,
            n_chroma=24,
        ).T

        if chromagram.shape[0] < speech_frames:
            chromagram = np.pad(
                chromagram, (0, speech_frames - chromagram.shape[0]), mode="edge"
            )
        else:
            chromagram = chromagram[:speech_frames]

        return chromagram

    def get_shifted_steps(self, src_wav_path, timbre_ref_wav_path):
        if src_wav_path == timbre_ref_wav_path:
            return 0

        src_f0 = extract_f0_hz(src_wav_path)
        timbre_ref_f0 = extract_f0_hz(timbre_ref_wav_path)

        src_f0_median = np.median(src_f0)
        timbre_ref_f0_median = np.median(timbre_ref_f0)

        src_shifted_steps = 12 * np.log2(timbre_ref_f0_median / src_f0_median)
        src_shifted_steps = round(src_shifted_steps)

        if src_shifted_steps > 12:
            src_shifted_steps = src_shifted_steps % 12
        elif src_shifted_steps < -12:
            src_shifted_steps = src_shifted_steps % -12

        return src_shifted_steps

    @torch.no_grad()
    def inference_fm(
        self,
        src_wav_path,
        timbre_ref_wav_path,
        src_wav_text="",
        timbre_ref_wav_text="",
        whisper_spec_perturb=False,
        use_pitch_shift=False,
        used_duration_of_timbre_ref_wav_path=None,
        flow_matching_steps=32,
        display_audio=False,
    ):
        src_speech, src_speech24k, src_speech16k = load_wav(src_wav_path, self.device)

        if display_audio:
            print("-" * 20)
            if src_wav_path == timbre_ref_wav_path:
                print("We want to reconstruct this audio:", src_wav_path)
                display_audio_in_notebook(src_wav_path, rate=24000)
            else:
                print("Source audio:")
                display_audio_in_notebook(src_speech, rate=24000)

        ## Whether to use shifted src to extract prosody and content-style ##
        if use_pitch_shift:
            src_shifted_steps = self.get_shifted_steps(
                src_wav_path, timbre_ref_wav_path
            )

            if display_audio:
                print("-" * 20)
                print(f"src_shifted_steps: {src_shifted_steps}")
        else:
            src_shifted_steps = 0

        ## Diffusion ##
        src_codecs = self.extract_coco_codec(
            "content_style",
            src_speech16k,
            src_speech,
            whisper_spec_perturb=whisper_spec_perturb,
            use_shifted_wav_to_extract_chromagram=use_pitch_shift,
            pitch_shift_steps=src_shifted_steps,
        )  # [1, T]

        predict_mel_feat = self.code2mel(
            src_codecs,
            timbre_ref_wav_path,
            prefix_text=timbre_ref_wav_text + " " + src_wav_text,
            used_duration_of_timbre_ref_wav_path=used_duration_of_timbre_ref_wav_path,
            flow_matching_steps=flow_matching_steps,
            logging=display_audio,
        )  # [1, T, D]

        ## Vocoder and Display ##
        synthesized_audio = self.mel2audio(
            predict_mel_feat, logging=display_audio
        )  # [1, T]

        return synthesized_audio

    def get_llm_prompt_text(self, text, follow_prosody_instruction, logging=False):
        llm_prompt_text = gen_chat_prompt(
            text,
            add_assistant_token=True,
            follow_prosody_instruction=follow_prosody_instruction,
        )

        # if logging:
        #     print("-" * 20)
        #     print("LLM Prompt Text:\n{}".format(llm_prompt_text))

        return llm_prompt_text

    def get_llm_prompt_prosody(
        self,
        use_prosody_code,
        predict_target_prosody,
        prosody_wav_path=None,
        style_ref_wav_path=None,
        use_pitch_shift=False,
        prosody_wav_pitch_shift_steps=0,
        style_ref_wav_pitch_shift_steps=0,
        target_duration=None,
        logging=False,
    ):
        if not use_prosody_code:
            return ""

        if not predict_target_prosody:
            # Just use the ground truth prosody #

            assert prosody_wav_path is not None
            prosody_speech, prosody_speech24k, prosody_speech16k = load_wav(
                prosody_wav_path, self.device
            )

            if target_duration is not None:
                # Calculate the chromagram frame len ratio
                prosody_wav_duration = prosody_speech.shape[0] / 24000
                prosody_wav_chromagram_frame_len_ratio = (
                    target_duration / prosody_wav_duration
                )
            else:
                prosody_wav_chromagram_frame_len_ratio = 1.0

            prosody_wav_prosody_ids = self.extract_coco_codec(
                "style",
                prosody_speech16k,
                prosody_speech,
                frame_len_ratio=prosody_wav_chromagram_frame_len_ratio,
                use_shifted_wav_to_extract_chromagram=use_pitch_shift,
                pitch_shift_steps=prosody_wav_pitch_shift_steps,
            )  # [1, T]

            if style_ref_wav_path is not None:
                style_ref_speech, style_ref_speech24k, style_ref_speech16k = load_wav(
                    style_ref_wav_path, self.device
                )
                style_ref_wav_prosody_ids = self.extract_coco_codec(
                    "style",
                    style_ref_speech16k,
                    style_ref_speech,
                    use_shifted_wav_to_extract_chromagram=use_pitch_shift,
                    pitch_shift_steps=style_ref_wav_pitch_shift_steps,
                )  # [1, T]
            else:
                style_ref_wav_prosody_ids = torch.zeros(1, 0).to(self.device)

            prosody_ids = torch.cat(
                [style_ref_wav_prosody_ids, prosody_wav_prosody_ids], dim=1
            )  # [1, T]

            prosody_ids = prosody_ids[0].tolist()
            prosody_text = "".join(["<|prosody_{}|>".format(i) for i in prosody_ids])
            prosody_text = "<|prosody_start|>" + prosody_text + "<|prosody_end|>"

            if logging:
                print("-" * 20)
                print("Prosody (Melody) Audio: ", prosody_wav_path)
                display_audio_in_notebook(prosody_speech, rate=24000)

        else:
            raise NotImplementedError("Not implemented yet")

        return prosody_text

    def get_llm_prompt_contentstyle(
        self,
        style_ref_wav_path=None,
        use_pitch_shift=False,
        pitch_shift_steps=0,
        logging=False,
    ):
        if style_ref_wav_path is None:
            return "<|content_style_start|>"

        style_ref_speech, style_ref_speech24k, style_ref_speech16k = load_wav(
            style_ref_wav_path, self.device
        )
        if logging:
            print("-" * 20)
            print("Style Reference Audio: ", style_ref_wav_path)
            display_audio_in_notebook(style_ref_speech, rate=24000)

        prompt_output_ids = self.extract_coco_codec(
            "content_style",
            style_ref_speech16k,
            style_ref_speech,
            use_shifted_wav_to_extract_chromagram=use_pitch_shift,
            pitch_shift_steps=pitch_shift_steps,
        )  # [1, T]

        prompt_output_ids = prompt_output_ids[0].tolist()
        prompt_output_text = "".join(
            ["<|content_style_{}|>".format(i) for i in prompt_output_ids]
        )
        prompt_output_text = "<|content_style_start|>" + prompt_output_text
        return prompt_output_text

    def parse_llm_generated_ids(self, generated_ids, llm_input_ids, logging=False):
        """
        Args:
            generated_ids: [1, T]
            llm_input_ids: [1, T]
        Returns:
            coco_codecs: [1, T]
        """
        input_len = llm_input_ids.shape[1]
        generated_ids = generated_ids[:, input_len:]

        # Eg: <|content_style_start|> <|content_style_1|> <|content_style_2|> <|content_style_end|> <|im_end|>
        generated_text = self.ar_tokenizer.decode(
            generated_ids[0], skip_special_tokens=False
        )

        content_style_ids = extract_special_ids(generated_text, prefix="content_style")
        content_style_ids = (
            torch.tensor(content_style_ids, dtype=torch.long)
            .to(self.device)
            .unsqueeze(0)
        )  # [1, T]

        if logging:
            print("-" * 20)
            print("LLM input_ids: ", llm_input_ids.shape)
            print("Generated content-style ids: ", content_style_ids.shape)

        return content_style_ids

    @torch.no_grad()
    def code2mel(
        self,
        contentstyle_codecs,
        timbre_ref_wav_path,
        prefix_text="",
        used_duration_of_timbre_ref_wav_path=None,
        flow_matching_steps=32,
        logging=False,
    ):
        timbre_ref_speech, timbre_ref_speech24k, timbre_ref_speech16k = load_wav(
            timbre_ref_wav_path,
            self.device,
            used_duration=used_duration_of_timbre_ref_wav_path,
        )
        if logging:
            print("-" * 20)
            print("Timbre Reference Audio: ", timbre_ref_wav_path)
            display_audio_in_notebook(timbre_ref_speech, rate=24000)

        timbre_ref_codecs = self.extract_coco_codec(
            "content_style",
            timbre_ref_speech16k,
            timbre_ref_speech,
        )  # [1, T]

        diffusion_input_codecs = torch.cat(
            [timbre_ref_codecs, contentstyle_codecs], dim=1
        )

        # Prepare the condition for diffusion
        diffusion_cond = self.fmt_model.cond_emb(diffusion_input_codecs)  # [1, T, D]
        if self.fmt_model.do_resampling:
            # Align to the frame rate of Mels
            diffusion_cond = self.fmt_model.resampling_layers(
                diffusion_cond.transpose(1, 2)
            ).transpose(1, 2)

        timbre_ref_mels = self.extract_mel_feature(timbre_ref_speech24k)  # [1, T, D]

        # Text as condition
        if self.fmt_use_text_as_condition:
            prefix_text_ids = self.fmt_text_tokenizer.encode(
                prefix_text, add_special_tokens=False
            )
            prefix_text_ids = torch.tensor(prefix_text_ids, dtype=torch.long).to(
                self.device
            )  # [T]
            prefix_text_ids = prefix_text_ids.unsqueeze(0)  # [1, T]
            prefix_text_embedding = self.fmt_model.text_cond_emb(
                prefix_text_ids
            )  # [1, T, D]
        else:
            prefix_text_embedding = None

        # [1, T, D]
        predict_mel_feat = self.fmt_model.reverse_diffusion(
            cond=diffusion_cond,
            prompt=timbre_ref_mels,
            text_embedding=prefix_text_embedding,
            n_timesteps=flow_matching_steps,
        )
        return predict_mel_feat

    @torch.no_grad()
    def mel2audio(self, predict_mel_feat, logging=False):
        # [1, 1, T] -> [1, T]
        synthesized_audio = (
            self.vocoder_model(predict_mel_feat.transpose(1, 2)).detach().cpu()
        )[0]

        if logging:
            print("-" * 20)
            print("Synthesized Audio:")
            # [T]
            audio = synthesized_audio.numpy()[0]
            display_audio_in_notebook(audio, rate=24000)

        return synthesized_audio

    @torch.no_grad()
    def inference_ar_and_fm(
        self,
        target_text=None,
        prosody_wav_path=None,
        style_ref_wav_path=None,
        style_ref_wav_text="",
        timbre_ref_wav_path=None,
        use_prosody_code=True,
        predict_target_prosody=False,
        top_k=25,
        top_p=0.8,
        temperature=1.0,
        use_pitch_shift=False,
        prosody_wav_shifted_steps=0,
        style_ref_wav_shifted_steps=0,
        target_duration=None,
        used_duration_of_timbre_ref_wav_path=None,
        flow_matching_steps=32,
        display_audio=False,
    ):
        """
        Based on the style reference wav to conduct the continuation generation:
            [Style_reference_Text, Target_Text], [Style_reference_Prosody, Target_Prosody], [Style_reference_cscodes, Target_cscodes]
        """
        assert self.ar_model is not None
        assert target_text is not None
        # assert style_ref_wav_path is not None
        assert timbre_ref_wav_path is not None

        ## Text Tokens ##
        if display_audio:
            print("-" * 20)
            print("Target text: \n", target_text)

        input_text = style_ref_wav_text + " " + target_text
        llm_prompt_text = self.get_llm_prompt_text(
            input_text, use_prosody_code, logging=display_audio
        )

        ## Whether to use shifted chromagram for timbre_ref_wav ##
        if use_pitch_shift:
            if prosody_wav_path is not None and prosody_wav_shifted_steps == 0:
                prosody_wav_shifted_steps = self.get_shifted_steps(
                    prosody_wav_path, timbre_ref_wav_path
                )
            if style_ref_wav_path is not None and style_ref_wav_shifted_steps == 0:
                style_ref_wav_shifted_steps = self.get_shifted_steps(
                    style_ref_wav_path, timbre_ref_wav_path
                )

            if display_audio:
                print("-" * 20)
                print(f"prosody_wav_shifted_steps: {prosody_wav_shifted_steps}")
                print(f"style_ref_wav_shifted_steps: {style_ref_wav_shifted_steps}")

        ## Prosody Tokens ##
        llm_prompt_prosody = self.get_llm_prompt_prosody(
            use_prosody_code,
            predict_target_prosody,
            prosody_wav_path=prosody_wav_path,
            style_ref_wav_path=style_ref_wav_path,
            use_pitch_shift=use_pitch_shift,
            prosody_wav_pitch_shift_steps=prosody_wav_shifted_steps,
            style_ref_wav_pitch_shift_steps=style_ref_wav_shifted_steps,
            target_duration=target_duration,
            logging=display_audio,
        )

        ## Content-Style Tokens ##
        llm_prompt_contentstyle = self.get_llm_prompt_contentstyle(
            style_ref_wav_path,
            use_pitch_shift=use_pitch_shift,
            pitch_shift_steps=style_ref_wav_shifted_steps,
            logging=display_audio,
        )

        ## AR ##
        llm_prompt = llm_prompt_text + llm_prompt_prosody + llm_prompt_contentstyle
        llm_input_ids = self.ar_tokenizer.encode(llm_prompt, add_special_tokens=True)
        llm_input_ids = (
            torch.tensor(llm_input_ids, dtype=torch.long).to(self.device).unsqueeze(0)
        )  # [1, T]

        if self.use_vllm:
            sampling_params = SamplingParams(
                max_tokens=500,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stop_token_ids=[self.ar_tokenizer.eos_token_id],
                skip_special_tokens=False,
            )
            outputs = self.ar_model.generate([llm_prompt], sampling_params)

            assert len(outputs) == 1
            output = outputs[0]
            prompt = output.prompt
            predicted_coco_text = output.outputs[0].text
            # print("vllm predicted_coco_text: ", predicted_coco_text)

            predicted_coco_codecs = extract_special_ids(
                predicted_coco_text, prefix="content_style"
            )
            predicted_coco_codecs = (
                torch.tensor(predicted_coco_codecs, dtype=torch.long)
                .to(self.device)
                .unsqueeze(0)
            )  # [1, T]
        else:
            generate_ids = self.ar_model.generate(
                input_ids=llm_input_ids,
                min_new_tokens=15,
                max_new_tokens=500,
                eos_token_id=self.ar_tokenizer.eos_token_id,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )  # [1, T]

            predicted_coco_codecs = self.parse_llm_generated_ids(
                generate_ids, llm_input_ids, logging=display_audio
            )  # [1, T]

        ## Diffusion ##
        predict_mel_feat = self.code2mel(
            predicted_coco_codecs,
            timbre_ref_wav_path,
            used_duration_of_timbre_ref_wav_path=used_duration_of_timbre_ref_wav_path,
            flow_matching_steps=flow_matching_steps,
            logging=display_audio,
        )  # [1, T, D]

        ## Vocoder ##
        synthesized_audio = self.mel2audio(
            predict_mel_feat, logging=display_audio
        )  # [1, T]
        return synthesized_audio

    @torch.no_grad()
    def inference_vocoder_resynthesis(self, wav_path, display_audio=False):
        speech, speech24k, speech16k = load_wav(wav_path, self.device)
        if display_audio:
            print("Ground Truth audio:")
            display_audio_in_notebook(speech, rate=24000)

        mel = self.extract_mel_feature(speech24k)  # [1, T, D]
        audio = self.vocoder_model(mel.transpose(1, 2)).detach().cpu()[0]
        if display_audio:
            print("Resynthesized audio:")
            display_audio_in_notebook(audio, rate=24000)
        return audio
