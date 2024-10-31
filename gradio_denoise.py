# 端口：7999
# 自动切分，可以加载长音频
import gradio as gr
import torch
import tempfile
import os
from pydub import AudioSegment
from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample


# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, df, _ = init_df("./DeepFilterNet2", config_allow_defaults=True)
model = model.to(device=device).eval()

def split_audio(audio_path, chunk_length_ms=600000):
    """将音频文件切割为指定长度的片段"""
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def merge_audio(chunks, output_path):
    """合并多个音频片段"""
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk
    combined.export(output_path, format="wav")

def process_chunk(chunk, index, temp_folder):
    """处理单个音频片段并返回处理后的路径"""
    chunk_path = os.path.join(temp_folder, f"chunk_{index}.wav")
    chunk.export(chunk_path, format="wav")
    output_path = os.path.join(temp_folder, f"processed_{index}.wav")
    denoise_audio(chunk_path, output_path)
    return output_path

def denoise_audio(input_audio_path: str, output_audio_path: str = None) -> str:
    """降噪音频文件"""
    print("开始降噪...")
    sr = 48000
    sample, meta = load_audio(input_audio_path, sr)
    if sample.dim() > 1 and sample.shape[0] > 1:
        sample = sample.mean(dim=0, keepdim=True)
    enhanced = enhance(model, df, sample)
    if meta.sample_rate != sr:
        enhanced = resample(enhanced, sr, meta.sample_rate)
    if output_audio_path is None:
        output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    save_audio(output_audio_path, enhanced, meta.sample_rate)
    return output_audio_path

def process_audio(input_file: str) -> str:
    """处理上传的音频文件，包括降噪和可能的格式转换"""
    temp_folder = tempfile.mkdtemp()
    output_audio_path = os.path.join(temp_folder, "final_output.wav")

    # 将音频切割成片段并处理每个片段
    chunks = split_audio(input_file)
    processed_chunks = [AudioSegment.from_wav(process_chunk(chunk, i, temp_folder)) for i, chunk in enumerate(chunks)]
    
    # 合并处理后的音频片段
    merge_audio(processed_chunks, output_audio_path)
    
    return output_audio_path

def gradio_interface(input_audio):
    """Gradio接口函数，处理音频并返回结果"""
    processed_audio_path = process_audio(input_audio)
    return processed_audio_path

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>录音降噪——JYD</h1>")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="上传需要降噪音频文件")
        process_button = gr.Button("降噪", variant="primary")
    with gr.Row():
        gr.Examples(
                [
                    "/opt/jyd01/wangruihua/data/audio/noise1.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise2.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise3.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise4.mp3",
                    "/opt/jyd01/wangruihua/data/audio/noise_class.wav",
                    # "/opt/jyd01/wangruihua/4090copy/synthesis/audio/101数学课.wav",
                    # "/home/jyd01/wangruihua/synthesis/audio/101历史课.wav",
                ],
                [audio_input],label='课堂音频')
    audio_output = gr.Audio(label="降噪后的音频")

    process_button.click(fn=gradio_interface, inputs=audio_input, outputs=audio_output)

if __name__ == "__main__":
    demo.launch(server_port=7999, server_name='0.0.0.0')
