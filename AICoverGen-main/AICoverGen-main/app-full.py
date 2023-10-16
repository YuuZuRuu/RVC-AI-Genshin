import os
import json
import argparse
import traceback
import logging
import gradio as gr
import numpy as np
import librosa
import torch
import asyncio
import edge_tts
import yt_dlp
import ffmpeg
import subprocess
import sys
import io
import wave
from datetime import datetime
from fairseq import checkpoint_utils
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from vc_infer_pipeline import VC
from config import (
    is_half,
    device
)
logging.getLogger("numba").setLevel(logging.WARNING)
limitation = os.getenv("SYSTEM") == "spaces"  # limit audio length in huggingface spaces

def create_vc_fn(tgt_sr, net_g, vc, if_f0, file_index, file_big_npy):
    def vc_fn(
        input_audio,
        upload_audio,
        upload_mode,
        f0_up_key,
        f0_method,
        index_rate,
        tts_mode,
        tts_text,
        tts_voice
    ):
        try:
            if tts_mode:
                if len(tts_text) > 100 and limitation:
                    return "Text is too long", None
                if tts_text is None or tts_voice is None:
                    return "You need to enter text and select a voice", None
                asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save("tts.mp3"))
                audio, sr = librosa.load("tts.mp3", sr=16000, mono=True)
            else:
                if upload_mode:
                    if input_audio is None:
                        return "You need to upload an audio", None
                    sampling_rate, audio = upload_audio
                    duration = audio.shape[0] / sampling_rate
                    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio.transpose(1, 0))
                    if sampling_rate != 16000:
                        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio, sr = librosa.load(input_audio, sr=16000, mono=True)
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                file_big_npy,
                index_rate,
                if_f0,
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            )
            return "Success", (tgt_sr, audio_opt)
        except:
            info = traceback.format_exc()
            print(info)
            return info, (None, None)
    return vc_fn

def cut_vocal_and_inst(yt_url):
    if yt_url != "":
        if not os.path.exists("youtube_audio"):
            os.mkdir("youtube_audio")
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            "outtmpl": 'youtube_audio/audio',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])
        yt_audio_path = "youtube_audio/audio.wav"
        command = f"demucs --two-stems=vocals {yt_audio_path}"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        print(result.stdout.decode())
        return ("separated/htdemucs/audio/vocals.wav", "separated/htdemucs/audio/no_vocals.wav", yt_audio_path, "separated/htdemucs/audio/vocals.wav")

def combine_vocal_and_inst(audio_data, audio_volume):
    print(audio_data)
    if not os.path.exists("result"):
        os.mkdir("result")
    vocal_path = "result/output.wav"
    inst_path = "separated/htdemucs/audio/no_vocals.wav"
    output_path = "result/combine.mp3"
    with wave.open(vocal_path, "w") as wave_file:
        wave_file.setnchannels(1) 
        wave_file.setsampwidth(2)
        wave_file.setframerate(audio_data[0])
        wave_file.writeframes(audio_data[1].tobytes())
    command =  f'ffmpeg -y -i {inst_path} -i {vocal_path} -filter_complex [1:a]volume={audio_volume}dB[v];[0:a][v]amix=inputs=2:duration=longest -b:a 320k -c:a libmp3lame {output_path}'
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    return output_path

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def change_to_tts_mode(tts_mode, upload_mode):
    if tts_mode:
        return gr.Textbox.update(visible=False), gr.Audio.update(visible=False), gr.Checkbox.update(visible=False), gr.Textbox.update(visible=True), gr.Dropdown.update(visible=True)
    else:
        if upload_mode:
            return gr.Textbox.update(visible=False), gr.Audio.update(visible=True), gr.Checkbox.update(visible=True), gr.Textbox.update(visible=False), gr.Dropdown.update(visible=False)
        else:
            return gr.Textbox.update(visible=True), gr.Audio.update(visible=False), gr.Checkbox.update(visible=True), gr.Textbox.update(visible=False), gr.Dropdown.update(visible=False)

def change_to_upload_mode(upload_mode):
    if upload_mode:
        return gr.Textbox().update(visible=False), gr.Audio().update(visible=True)
    else:
        return gr.Textbox().update(visible=True), gr.Audio().update(visible=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--colab", action="store_true", default=False, help="share gradio app")
    args, unknown = parser.parse_known_args()
    load_hubert()
    models = []
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
    with open("weights/model_info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)
    for name, info in models_info.items():
        if not info['enable']:
            continue
        title = info['title']
        author = info.get("author", None)
        cover = f"weights/{name}/{info['cover']}"
        index = f"weights/{name}/{info['feature_retrieval_library']}"
        npy = f"weights/{name}/{info['feature_file']}"
        cpt = torch.load(f"weights/{name}/{name}.pth", map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净, 真奇葩
        net_g.eval().to(device)
        if is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, device, is_half)
        models.append((name, title, author, cover, create_vc_fn(tgt_sr, net_g, vc, if_f0, index, npy)))
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> RVC Models\n"
            "## <center> The input audio should be clean and pure voice without background music.\n"
            "### <center> More feature will be added soon... \n"
            "[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hx6kKvIuv5XNY1Gai2PEuZhpO5z6xpVh?usp=sharing)\n\n"
            "[![Original Repo](https://badgen.net/badge/icon/github?icon=github&label=Original%20Repo)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)"
        )
        with gr.Tabs():
            for (name, title, author, cover, vc_fn) in models:
                with gr.TabItem(name):
                    with gr.Row():
                        gr.Markdown(
                            '<div align="center">'
                            f'<div>{title}</div>\n'+
                            (f'<div>Model author: {author}</div>' if author else "")+
                            (f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else "")+
                            '</div>'
                        )
                    with gr.Row():
                        with gr.Column():
                            vc_youtube = gr.Textbox(label="Youtube URL")
                            vc_convert = gr.Button("Convert", variant="primary")
                            vc_vocal_preview = gr.Audio(label="Vocal Preview")
                            vc_inst_preview = gr.Audio(label="Instrumental Preview")
                            vc_audio_preview = gr.Audio(label="Audio Preview")
                        with gr.Column():
                            vc_input = gr.Textbox(label="Input audio path")
                            vc_upload = gr.Audio(label="Upload audio file", visible=False, interactive=True)
                            upload_mode = gr.Checkbox(label="Upload mode", value=False)
                            vc_transpose = gr.Number(label="Transpose", value=0)
                            vc_f0method = gr.Radio(
                                label="Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies",
                                choices=["pm", "harvest"],
                                value="pm",
                                interactive=True,
                            )
                            vc_index_ratio = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Retrieval feature ratio",
                                value=0.6,
                                interactive=True,
                            )
                            tts_mode = gr.Checkbox(label="tts (use edge-tts as input)", value=False)
                            tts_text = gr.Textbox(visible=False,label="TTS text (100 words limitation)" if limitation else "TTS text")
                            tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female")
                            vc_output1 = gr.Textbox(label="Output Message")
                            vc_output2 = gr.Audio(label="Output Audio")
                            vc_submit = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            vc_volume = gr.Slider(
                                minimum=0,
                                maximum=10,
                                label="Vocal volume",
                                value=4,
                                interactive=True,
                                step=1
                            )
                            vc_outputCombine = gr.Audio(label="Output Combined Audio")
                            vc_combine =  gr.Button("Combine",variant="primary")
                vc_submit.click(vc_fn, [vc_input, vc_upload, upload_mode, vc_transpose, vc_f0method, vc_index_ratio, tts_mode, tts_text, tts_voice], [vc_output1, vc_output2])
                vc_convert.click(cut_vocal_and_inst, vc_youtube, [vc_vocal_preview, vc_inst_preview, vc_audio_preview, vc_input])
                vc_combine.click(combine_vocal_and_inst, [vc_output2, vc_volume], vc_outputCombine)
                tts_mode.change(change_to_tts_mode, [tts_mode, upload_mode], [vc_input, vc_upload, upload_mode, tts_text, tts_voice])
                upload_mode.change(change_to_upload_mode, [upload_mode], [vc_input, vc_upload])
        app.queue(concurrency_count=1, max_size=20, api_open=args.api).launch(share=args.colab)