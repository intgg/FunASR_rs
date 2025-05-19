"""
高效加载版实时语音识别 Demo - FunASR
----------------------------
本程序采用FunASR.py的高效模型加载方式，同时保留了VAD和标点恢复功能。
通过延迟加载和预缓存策略，大幅提高了启动速度。

使用方法:
- 按回车开始录音
- 对着麦克风说话
- 再次按回车停止录音

依赖库:
- funasr
- sounddevice
- numpy
"""

from funasr import AutoModel
import sounddevice as sd
import numpy as np
import threading
import time
import queue
import os


class FastLoadASR:
    def __init__(self, use_vad=True, use_punc=True, disable_update=True):
        """
        初始化快速加载版语音识别系统

        参数:
            use_vad: 是否使用语音端点检测
            use_punc: 是否使用标点恢复
            disable_update: 是否禁用FunASR更新检查(加速启动)
        """
        # 功能开关
        self.use_vad = use_vad
        self.use_punc = use_punc
        self.disable_update = disable_update

        # 语音识别参数设置
        self.sample_rate = 16000  # 采样率(Hz)

        # ASR参数
        self.asr_chunk_size = [0, 10, 5]  # 流式设置：[0, 10, 5] = 600ms
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        # VAD参数
        self.vad_chunk_duration_ms = 200  # VAD每个音频块的持续时间(毫秒)
        self.vad_chunk_samples = int(self.sample_rate * self.vad_chunk_duration_ms / 1000)

        # ASR参数
        self.asr_chunk_duration_ms = 600  # 每个ASR音频块的持续时间(毫秒)
        self.asr_chunk_samples = int(self.sample_rate * self.asr_chunk_duration_ms / 1000)

        # 运行时变量
        self.running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.complete_transcript = ""
        self.raw_transcript = ""
        self.is_speaking = False
        self.speech_buffer = np.array([], dtype=np.float32)

        # 模型变量
        self.asr_model = None
        self.vad_model = None
        self.punc_model = None
        self.vad_cache = {}
        self.asr_cache = {}

        # 设置环境变量以加快加载
        if self.disable_update:
            os.environ["FUNASR_DISABLE_UPDATE"] = "True"

        # 异步预加载ASR模型
        print("开始加载ASR模型...")
        self.asr_load_thread = threading.Thread(target=self.load_asr_model)
        self.asr_load_thread.daemon = True
        self.asr_load_thread.start()

    def load_asr_model(self):
        """加载ASR模型的线程函数"""
        try:
            # 使用与FunASR.py相同的加载方式
            self.asr_model = AutoModel(model="paraformer-zh-streaming")
            print("ASR模型加载完成!")
        except Exception as e:
            print(f"ASR模型加载失败: {e}")

    def ensure_asr_model_loaded(self):
        """确保ASR模型已加载"""
        if self.asr_model is None:
            print("等待ASR模型加载完成...")
            if hasattr(self, 'asr_load_thread'):
                self.asr_load_thread.join()

            # 如果线程结束后模型仍未加载，再次尝试加载
            if self.asr_model is None:
                print("重新尝试加载ASR模型...")
                try:
                    self.asr_model = AutoModel(model="paraformer-zh-streaming")
                    print("ASR模型加载完成!")
                except Exception as e:
                    print(f"ASR模型加载失败: {e}")
                    return False
        return True

    def load_vad_model_if_needed(self):
        """仅在需要时加载VAD模型"""
        if self.use_vad and self.vad_model is None:
            print("加载VAD模型...")
            try:
                self.vad_model = AutoModel(model="fsmn-vad")
                print("VAD模型加载完成!")
                return True
            except Exception as e:
                print(f"VAD模型加载失败: {e}")
                return False
        return True

    def load_punc_model_if_needed(self):
        """仅在需要时加载标点恢复模型"""
        if self.use_punc and self.punc_model is None:
            print("加载标点恢复模型...")
            try:
                self.punc_model = AutoModel(model="ct-punc")
                print("标点恢复模型加载完成!")
                return True
            except Exception as e:
                print(f"标点恢复模型加载失败: {e}")
                return False
        return True

    def audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            print(f"音频状态: {status}")
        # 将音频数据放入队列
        self.audio_queue.put(indata.copy())

    def process_audio_thread(self):
        """音频处理线程"""
        vad_buffer = np.array([], dtype=np.float32)

        while self.running:
            try:
                # 从队列获取音频数据
                while not self.audio_queue.empty() and self.running:
                    chunk = self.audio_queue.get_nowait()
                    if self.use_vad:
                        vad_buffer = np.append(vad_buffer, chunk.flatten())
                    else:
                        # 不使用VAD时，直接将音频块添加到语音缓冲区
                        self.speech_buffer = np.append(self.speech_buffer, chunk.flatten())

                # 使用VAD处理
                if self.use_vad and self.vad_model is not None:
                    while len(vad_buffer) >= self.vad_chunk_samples and self.running:
                        # 提取一个VAD音频块
                        vad_chunk = vad_buffer[:self.vad_chunk_samples]
                        vad_buffer = vad_buffer[self.vad_chunk_samples:]

                        # 使用VAD模型处理
                        vad_res = self.vad_model.generate(
                            input=vad_chunk,
                            cache=self.vad_cache,
                            is_final=False,
                            chunk_size=self.vad_chunk_duration_ms
                        )

                        # 处理VAD结果
                        if len(vad_res[0]["value"]):
                            # 有语音活动检测结果
                            for segment in vad_res[0]["value"]:
                                if segment[0] != -1 and segment[1] == -1:
                                    # 检测到语音开始
                                    self.is_speaking = True
                                    print("\n检测到语音开始...")
                                elif segment[0] == -1 and segment[1] != -1:
                                    # 检测到语音结束
                                    self.is_speaking = False
                                    print("\n检测到语音结束...")
                                    # 处理积累的语音缓冲区
                                    if len(self.speech_buffer) > 0:
                                        self.process_asr_buffer(is_final=True)

                        # 如果正在说话，将当前块添加到语音缓冲区
                        if self.is_speaking:
                            self.speech_buffer = np.append(self.speech_buffer, vad_chunk)
                else:
                    # 不使用VAD时，总是处于"说话"状态
                    self.is_speaking = True

                # 如果语音缓冲区足够大，进行ASR处理
                if len(self.speech_buffer) >= self.asr_chunk_samples:
                    self.process_asr_buffer()

                # 短暂休眠以减少CPU使用
                time.sleep(0.01)
            except Exception as e:
                print(f"\n音频处理错误: {e}")

    def process_asr_buffer(self, is_final=False):
        """处理语音缓冲区进行ASR识别"""
        if self.asr_model is None:
            return

        try:
            # 如果没有足够的样本而且不是最终处理，则返回
            if len(self.speech_buffer) < self.asr_chunk_samples and not is_final:
                return

            # 如果不是最终处理，提取一个ASR块
            if not is_final:
                asr_chunk = self.speech_buffer[:self.asr_chunk_samples]
                self.speech_buffer = self.speech_buffer[self.asr_chunk_samples:]
            else:
                # 如果是最终处理，使用整个缓冲区
                asr_chunk = self.speech_buffer
                self.speech_buffer = np.array([], dtype=np.float32)

            # 使用ASR模型处理
            if len(asr_chunk) > 0:
                asr_res = self.asr_model.generate(
                    input=asr_chunk,
                    cache=self.asr_cache,
                    is_final=is_final,
                    chunk_size=self.asr_chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )

                # 如果有识别结果，处理并应用标点
                if asr_res[0]["text"]:
                    text = asr_res[0]["text"]
                    self.raw_transcript += text

                    # 应用标点恢复 (如果启用)
                    if self.use_punc and self.punc_model is not None:
                        punc_res = self.punc_model.generate(input=self.raw_transcript)
                        if punc_res:
                            punctuated_text = punc_res[0]["text"]
                            # 更新完整转写并添加到结果队列
                            self.complete_transcript = punctuated_text
                            self.result_queue.put((text, punctuated_text))
                    else:
                        # 不使用标点恢复时，直接使用原始文本
                        self.complete_transcript = self.raw_transcript
                        self.result_queue.put((text, self.raw_transcript))
        except Exception as e:
            print(f"\nASR处理错误: {e}")

    def display_thread(self):
        """结果显示线程"""
        while self.running:
            try:
                # 检查新结果
                while not self.result_queue.empty() and self.running:
                    raw_text, punctuated_text = self.result_queue.get_nowait()
                    print(f"\r新识别内容: {raw_text}")
                    print(f"完整转写: {punctuated_text}")
                    print("-" * 50)

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"\n显示结果错误: {e}")

    def start(self):
        """开始语音识别"""
        if self.running:
            return

        # 确保ASR模型已加载
        if not self.ensure_asr_model_loaded():
            print("无法启动语音识别：ASR模型加载失败")
            return

        # 根据需要加载其他模型
        if self.use_vad:
            self.load_vad_model_if_needed()

        if self.use_punc:
            self.load_punc_model_if_needed()

        # 重置状态变量
        self.running = True
        self.vad_cache = {}
        self.asr_cache = {}
        self.complete_transcript = ""
        self.raw_transcript = ""
        self.is_speaking = not self.use_vad  # 不使用VAD时默认为说话状态
        self.speech_buffer = np.array([], dtype=np.float32)

        # 启动处理线程
        self.process_thread = threading.Thread(target=self.process_audio_thread)
        self.process_thread.daemon = True
        self.process_thread.start()

        # 启动显示线程
        self.display_thread = threading.Thread(target=self.display_thread)
        self.display_thread.daemon = True
        self.display_thread.start()

        # 启动音频流
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms音频块
        )
        self.stream.start()

        # 显示启动状态
        features = []
        if self.use_vad and self.vad_model is not None:
            features.append("语音端点检测")
        features.append("语音识别")
        if self.use_punc and self.punc_model is not None:
            features.append("标点恢复")

        print("\n🎤 语音识别已启动，包含" + "、".join(features) + "功能")
        print("请对着麦克风说话...")
        print("-" * 50)

    def stop(self):
        """停止语音识别"""
        if not self.running:
            return

        self.running = False

        # 停止音频流
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        # 等待线程结束
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
        if hasattr(self, 'display_thread'):
            self.display_thread.join(timeout=2.0)

        # 处理最终剩余的音频
        try:
            # 最终VAD处理
            if self.use_vad and self.vad_model is not None:
                self.vad_model.generate(
                    input=np.zeros(1, dtype=np.float32),
                    cache=self.vad_cache,
                    is_final=True,
                    chunk_size=self.vad_chunk_duration_ms
                )

            # 处理剩余的语音缓冲区
            if len(self.speech_buffer) > 0:
                self.process_asr_buffer(is_final=True)

            # 最终ASR处理，强制输出最后的文字
            if self.asr_model is not None:
                self.asr_model.generate(
                    input=np.zeros(1, dtype=np.float32),
                    cache=self.asr_cache,
                    is_final=True,
                    chunk_size=self.asr_chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )

            # 对最终文本应用标点恢复
            if self.raw_transcript and self.use_punc and self.punc_model is not None:
                final_punc_res = self.punc_model.generate(input=self.raw_transcript)
                if final_punc_res:
                    self.complete_transcript = final_punc_res[0]["text"]
        except Exception as e:
            print(f"\n最终处理错误: {e}")

        print("-" * 50)
        print("🛑 语音识别已停止。")
        print("\n最终转写结果:")
        print(self.complete_transcript)
        print("-" * 50)


def main():
    """主函数"""
    print("=" * 60)
    print("高效加载版实时语音识别 Demo - FunASR")
    print("=" * 60)

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="FunASR高效加载实时语音识别Demo")
    parser.add_argument("--no-vad", action="store_true", help="禁用语音端点检测")
    parser.add_argument("--no-punc", action="store_true", help="禁用标点恢复")
    parser.add_argument("--check-update", action="store_true", help="启用FunASR更新检查")
    args = parser.parse_args()

    # 创建语音识别实例
    asr = FastLoadASR(
        use_vad=not args.no_vad,
        use_punc=not args.no_punc,
        disable_update=not args.check_update
    )

    # 显示功能状态
    print(f"语音端点检测: {'已禁用' if args.no_vad else '已启用'}")
    print(f"标点恢复: {'已禁用' if args.no_punc else '已启用'}")
    print(f"FunASR更新检查: {'已启用' if args.check_update else '已禁用'}")

    try:
        print("\n按回车键开始录音...")
        input()

        # 启动语音识别
        asr.start()

        print("\n按回车键停止录音...")
        input()

        # 停止语音识别
        asr.stop()

        print("\nDemo完成，感谢使用FunASR!")

    except KeyboardInterrupt:
        print("\n\n用户中断。正在停止...")
        if hasattr(asr, 'running') and asr.running:
            asr.stop()
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()