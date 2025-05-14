# funasr_module.py - FunASR离线语音识别模块（性能优化版）
import os
import sys
import wave
import time
import numpy as np
import pyaudio
import threading
from queue import Queue
from collections import deque
from io import BytesIO
import gc

try:
    from funasr import AutoModel

    print("FunASR模块加载成功")
except ImportError:
    print("错误: 未找到FunASR模块，请安装: pip install -U funasr")
    sys.exit(1)


class FunASRModule:
    """使用FunASR的实时语音识别模块 - 性能优化版"""

    # 使用__slots__减少内存占用
    __slots__ = ['model', 'RATE', 'CHUNK', 'FORMAT', 'CHANNELS', 'input_device_index',
                 'audio', 'is_running', 'is_recording', 'result_queue', 'record_thread',
                 'vad_threshold', 'vad_active_frames', 'vad_active_threshold',
                 'recording_active', 'speech_frames', 'min_valid_frames',
                 'max_silence_in_speech', 'continuous_silence', 'frame_status_queue',
                 'status_queue_size', 'allow_recognition', 'debug', 'stream',
                 '_cached_audio_array', '_cached_frames_count', 'last_gc_time']

    def __init__(self, app_id=None, api_key=None, input_device_index=None):
        """初始化FunASR模块，使用固定阈值500，支持选择输入设备"""
        # app_id和api_key参数仅为接口兼容性，FunASR不需要这些参数
        print("正在初始化FunASR模型...")

        # 初始化模型
        try:
            self.model = AutoModel(model="paraformer-zh", disable_update=True)
            print("FunASR模型初始化成功")
        except Exception as e:
            print(f"初始化FunASR模型失败: {e}")
            sys.exit(1)

        # 音频参数
        self.RATE = 16000  # 采样率
        self.CHUNK = 1024  # 数据块大小
        self.FORMAT = pyaudio.paInt16  # 16位格式
        self.CHANNELS = 1  # 单声道
        self.input_device_index = input_device_index  # 输入设备索引

        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()

        # 流对象，延迟初始化
        self.stream = None

        # 控制标志
        self.is_running = False
        self.is_recording = False

        # 结果队列和线程
        self.result_queue = Queue()
        self.record_thread = None

        # 设置固定阈值为500
        self.vad_threshold = 500
        print(f"语音检测阈值固定为: {self.vad_threshold}")

        # VAD参数
        self.vad_active_frames = 0
        self.vad_active_threshold = 3  # 连续帧数

        # 状态控制
        self.recording_active = False
        self.speech_frames = []
        self.min_valid_frames = 8  # 最小有效帧数
        self.max_silence_in_speech = 20  # 允许的最大静音帧数
        self.continuous_silence = 0

        # 使用deque优化帧状态队列
        self.status_queue_size = 3
        self.frame_status_queue = deque(maxlen=self.status_queue_size)

        # 是否允许识别
        self.allow_recognition = True

        # 调试模式
        self.debug = True

        # 缓存变量，用于减少重复计算
        self._cached_audio_array = None
        self._cached_frames_count = 0

        # 垃圾回收控制
        self.last_gc_time = time.time()

    def update_device(self, input_device_index):
        """更新输入设备（热切换）"""
        if self.is_running:
            print("警告: 设备更新时应先停止录音")
            return False

        self.input_device_index = input_device_index
        print(f"输入设备已更新为索引: {input_device_index}")
        return True

    def start(self):
        """启动实时语音识别"""
        if self.is_running:
            return

        self.is_running = True
        self.is_recording = True

        # 清空缓存
        self._cached_audio_array = None
        self._cached_frames_count = 0

        # 启动录音和识别线程
        self.record_thread = threading.Thread(target=self.record_and_recognize)
        self.record_thread.daemon = True
        self.record_thread.start()

        print(f"FunASR语音识别已启动，阈值固定为: {self.vad_threshold}")

    def stop(self):
        """停止实时语音识别"""
        self.is_running = False
        self.is_recording = False

        # 清空结果队列
        while not self.result_queue.empty():
            self.result_queue.get()

        # 等待线程结束
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)

        # 关闭音频流
        self._close_stream()

        print("FunASR语音识别已停止")

    def get_result(self):
        """获取识别结果"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def is_voice(self, audio_data):
        """检测帧是否包含语音"""
        # 转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # 使用numpy高效计算音频特征
        amplitude = np.abs(audio_array).max()

        # 检查是否超过阈值
        is_above_threshold = amplitude > self.vad_threshold

        # 调试输出
        if self.debug and is_above_threshold:
            print(f"检测到可能的语音: 振幅={amplitude}, 阈值={self.vad_threshold}")

        # 使用deque高效管理队列
        self.frame_status_queue.append(is_above_threshold)

        # 基于队列中的多帧判断是否为语音 (sum比循环更高效)
        return sum(self.frame_status_queue) >= 1  # 只要有一帧超过阈值就算语音

    def save_wav(self, frames, filename):
        """保存音频为WAV文件 - 仅用于调试"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))

        if self.debug:
            print(f"音频已保存到 {filename}, 数据长度: {len(frames)} 帧")

    def get_audio_array(self, frames):
        """获取音频数组，支持缓存"""
        frames_count = len(frames)

        # 使用缓存避免重复计算
        if hasattr(self, '_cached_audio_array') and self._cached_frames_count == frames_count:
            return self._cached_audio_array

        # 计算新的音频数组
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # 更新缓存
        self._cached_audio_array = audio_array
        self._cached_frames_count = frames_count

        return audio_array

    def should_perform_recognition(self, frames):
        """判断是否应该进行识别"""
        # 检查帧数是否足够
        frames_count = len(frames)
        if frames_count < self.min_valid_frames:
            if self.debug:
                print(f"录音太短 ({frames_count}帧) - 跳过识别")
            return False

        # 获取音频数组（使用缓存避免重复计算）
        audio_array = self.get_audio_array(frames)

        # 检查录音能量是否足够
        amplitude = np.abs(audio_array).max()

        # 使用宽松的标准
        threshold = self.vad_threshold * 0.7
        if amplitude < threshold:
            if self.debug:
                print(f"录音振幅不足 (振幅: {amplitude}, 阈值: {threshold}) - 跳过识别")
            return False

        # 通过所有检查
        if self.debug:
            print(f"录音检查通过 - 振幅: {amplitude}, 帧数: {frames_count}")

        return True

    def _open_stream(self):
        """打开音频流，仅在需要时初始化"""
        if self.stream is None or not self.stream.is_active():
            # 关闭旧流（如果存在）
            self._close_stream()

            # 创建新流
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.CHUNK
            )

        return self.stream

    def _close_stream(self):
        """安全关闭音频流"""
        if self.stream is not None:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"关闭音频流时出错: {e}")
            finally:
                self.stream = None

    def recognize_audio(self, frames):
        """优化的音频识别方法，直接使用内存中的数据"""
        try:
            # 直接使用内存中的数据进行识别，避免文件I/O
            audio_data = b''.join(frames)

            # 尝试使用内存流（如果FunASR支持）
            try:
                buffer = BytesIO()
                with wave.open(buffer, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(audio_data)

                # 将缓冲区指针移到开始位置
                buffer.seek(0)

                # 使用内存流进行识别
                result = self.model.generate(input=buffer)

            except Exception as e:
                # 回退到临时文件方案
                if self.debug:
                    print(f"内存识别失败，回退到文件方式: {e}")

                # 使用临时文件名
                temp_file = "temp_audio.wav"

                # 保存并识别
                self.save_wav(frames, temp_file)
                result = self.model.generate(input=temp_file)

                # 删除临时文件
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

            # 解析结果
            text = ""
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'text' in result[0]:
                    text = result[0]['text']
            elif isinstance(result, dict) and "text" in result:
                text = result["text"]
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)

            return text.strip()

        except Exception as e:
            print(f"识别过程中出错: {e}")
            return ""

    def record_and_recognize(self):
        """录音并识别（线程函数）"""
        # 打开音频流
        stream = self._open_stream()

        print("开始录音和识别...")

        # 上一次识别的文本
        last_text = ""
        gc_interval = 60  # 每60秒触发一次垃圾回收

        try:
            while self.is_running:
                # 定期触发垃圾回收
                current_time = time.time()
                if current_time - self.last_gc_time > gc_interval:
                    gc.collect()
                    self.last_gc_time = current_time

                # 检查是否正在录音
                if not self.is_recording or not self.allow_recognition:
                    time.sleep(0.1)
                    continue

                # 读取音频数据
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"读取音频数据错误: {e}")
                    continue

                # 检测是否是语音
                voice_detected = self.is_voice(data)

                # 更新状态
                if voice_detected:
                    self.vad_active_frames += 1
                    self.continuous_silence = 0
                else:
                    self.continuous_silence += 1
                    if not self.recording_active:
                        self.vad_active_frames = max(0, self.vad_active_frames - 1)

                # 语音开始
                if not self.recording_active and self.vad_active_frames >= self.vad_active_threshold:
                    self.recording_active = True
                    self.speech_frames = []
                    print("检测到语音 - 开始录制")

                # 录制语音
                if self.recording_active:
                    self.speech_frames.append(data)
                    if self.debug and len(self.speech_frames) % 10 == 0:
                        print(f"正在录制: 已收集 {len(self.speech_frames)} 帧")

                # 语音结束
                if self.recording_active and self.continuous_silence >= self.max_silence_in_speech:
                    self.recording_active = False
                    self.vad_active_frames = 0
                    print(f"检测到语音结束 - 停止录制，共 {len(self.speech_frames)} 帧")

                    # 检查是否应该进行识别
                    if self.should_perform_recognition(self.speech_frames):
                        # 临时禁用识别，避免在处理时收到新数据
                        self.allow_recognition = False

                        # 识别语音（优化后的方法，直接使用内存数据）
                        print("识别中...")
                        text = self.recognize_audio(self.speech_frames)

                        # 检查是否有有效文本
                        if text:
                            print(f"识别文本: {text}")
                            self.result_queue.put({"text": text, "is_final": False})

                            if text != last_text:
                                last_text = text
                                self.result_queue.put({"text": text, "is_final": True})
                            else:
                                print("识别结果与上次相同 - 跳过最终结果")
                        else:
                            print("识别结果为空 - 跳过")

                        # 恢复识别
                        self.allow_recognition = True

                    # 清空语音帧和缓存
                    self.speech_frames = []
                    self._cached_audio_array = None
                    self._cached_frames_count = 0

                # 短暂暂停
                time.sleep(0.01)

        except Exception as e:
            error_info = f"录音过程出错: {str(e)}"
            print(error_info)
            self.result_queue.put({"error": error_info})

        finally:
            # 关闭流
            self._close_stream()

    def cleanup(self):
        """清理资源"""
        self.stop()
        self.audio.terminate()
        print("FunASR资源已释放")


# 测试代码
if __name__ == "__main__":
    asr = FunASRModule()

    try:
        # 启动识别
        asr.start()

        # 持续获取结果
        print("\n开始实时语音识别，按Ctrl+C停止...")
        while True:
            result = asr.get_result()
            if result:
                if isinstance(result, dict):
                    if "error" in result:
                        print(f"\n错误: {result['error']}")
                        break
                    elif "text" in result:
                        if result["is_final"]:
                            print(f"\n[最终] {result['text']}")
                        else:
                            print(f"\r[中间] {result['text']}", end="")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n停止识别...")
        asr.stop()
        asr.cleanup()