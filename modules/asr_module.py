import sounddevice as sd
import soundfile as sf
import queue
import time
from funasr import AutoModel

# 初始化模型
asr_model = AutoModel(model="paraformer-zh-streaming")
punc_model = AutoModel(model="ct-punc")
vad_model = AutoModel(model="fsmn-vad")

# 参数
sample_rate = 16000
chunk_duration = 0.6
chunk_size = int(sample_rate * chunk_duration)

N = 15  # 触发标点的最小字数
T = 2   # 超过T秒无识别更新，则触发标点

chunk_conf = [0, 10, 5]
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1

# 状态变量
audio_queue = queue.Queue()
text_buffer = ""
last_recognize_time = time.time()
asr_cache = {}

# ✅ 最终识别结果累计输出
final_output = ""

def callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def main():
    global text_buffer, last_recognize_time, final_output

    print("开始实时语音识别... 按 Ctrl+C 停止")
    with sd.InputStream(channels=1, samplerate=sample_rate, blocksize=chunk_size,
                        dtype='float32', callback=callback):
        while True:
            audio_data = audio_queue.get()
            audio_data = audio_data.reshape(-1)

            # VAD 检测是否静音
            vad_result = vad_model.generate(input=audio_data)
            is_final = vad_result and isinstance(vad_result[0], list) and vad_result[0] and vad_result[0][0] == -1

            # 语音识别
            result = asr_model.generate(
                input=audio_data,
                cache=asr_cache,
                is_final=is_final,
                chunk_size=chunk_conf,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back
            )

            if isinstance(result, list) and len(result) > 0:
                new_text = result[0]['text']
                if new_text:
                    text_buffer += new_text
                    last_recognize_time = time.time()
                    print("识别追加:", text_buffer)

            # 是否触发标点处理
            now = time.time()
            if len(text_buffer) >= N or (now - last_recognize_time > T) or is_final:
                if text_buffer.strip():
                    punctuated = punc_model.generate(input=text_buffer.strip())[0]['text']
                    final_output += punctuated + " "  # ✅ 累积
                    print("[标点处理后]:", punctuated)
                    print("[当前完整识别]:", final_output.strip())
                text_buffer = ""

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n语音识别结束。")
        print("\n=== 完整识别内容 ===")
        print(final_output.strip())
