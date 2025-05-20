"""
实时语音识别与翻译系统 - GUI版本
---------------------------
这个版本添加了简单的GUI界面，使得操作更加直观。

使用方法:
- 从下拉菜单选择目标翻译语言
- 点击"开始录音"按钮开始录音
- 对着麦克风说话
- 点击"停止录音"按钮停止录音

依赖库:
- funasr
- sounddevice
- numpy
- requests
- tkinter (Python标准库)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import queue
import os
import sys
from datetime import datetime
import numpy as np
import json
import requests
import base64
import hmac
import hashlib
from time import mktime
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode
from threading import Lock

# 尝试导入语音识别相关库
try:
    from funasr import AutoModel
    import sounddevice as sd
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("警告: funasr或sounddevice未安装，语音识别功能将无法使用")

# 全局日志队列
log_queue = queue.Queue()

def log(message, show_time=True):
    """添加日志消息到队列"""
    if show_time:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        full_message = f"[{timestamp}] {message}"
    else:
        full_message = message
    log_queue.put(full_message)
    print(full_message)


#----------------------------------------
# 语音识别模块 (简化版)
#----------------------------------------
class FastLoadASR:
    def __init__(self, use_vad=True, use_punc=True, disable_update=True):
        """初始化快速加载版语音识别系统"""
        log("初始化语音识别模块...")

        if not FUNASR_AVAILABLE:
            log("错误: FunASR库未安装")
            return

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
        log("开始异步加载ASR模型...")
        self.asr_load_thread = threading.Thread(target=self.load_asr_model)
        self.asr_load_thread.daemon = True
        self.asr_load_thread.start()

    def load_asr_model(self):
        """加载ASR模型的线程函数"""
        try:
            log("正在加载ASR模型...")
            self.asr_model = AutoModel(model="paraformer-zh-streaming")
            log("ASR模型加载完成!")
        except Exception as e:
            log(f"ASR模型加载失败: {e}")

    def ensure_asr_model_loaded(self):
        """确保ASR模型已加载"""
        if self.asr_model is None:
            log("等待ASR模型加载完成...")
            if hasattr(self, 'asr_load_thread'):
                self.asr_load_thread.join()

            # 如果线程结束后模型仍未加载，再次尝试加载
            if self.asr_model is None:
                log("重新尝试加载ASR模型...")
                try:
                    self.asr_model = AutoModel(model="paraformer-zh-streaming")
                    log("ASR模型加载完成!")
                except Exception as e:
                    log(f"ASR模型加载失败: {e}")
                    return False
        return True

    def load_vad_model_if_needed(self):
        """仅在需要时加载VAD模型"""
        if self.use_vad and self.vad_model is None:
            log("加载VAD模型...")
            try:
                self.vad_model = AutoModel(model="fsmn-vad")
                log("VAD模型加载完成!")
                return True
            except Exception as e:
                log(f"VAD模型加载失败: {e}")
                return False
        return True

    def load_punc_model_if_needed(self):
        """仅在需要时加载标点恢复模型"""
        if self.use_punc and self.punc_model is None:
            log("加载标点恢复模型...")
            try:
                self.punc_model = AutoModel(model="ct-punc")
                log("标点恢复模型加载完成!")
                return True
            except Exception as e:
                log(f"标点恢复模型加载失败: {e}")
                return False
        return True

    def audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            log(f"音频状态: {status}")
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
                                    log("检测到语音开始...")
                                elif segment[0] == -1 and segment[1] != -1:
                                    # 检测到语音结束
                                    self.is_speaking = False
                                    log("检测到语音结束...")
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
                log(f"音频处理错误: {e}")

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
            log(f"ASR处理错误: {e}")

    def start(self):
        """开始语音识别"""
        if self.running:
            return False

        # 确保ASR模型已加载
        if not self.ensure_asr_model_loaded():
            log("无法启动语音识别：ASR模型加载失败")
            return False

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

        # 启动音频流
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms音频块
            )
            self.stream.start()
        except Exception as e:
            log(f"启动音频流失败: {e}")
            self.running = False
            return False

        # 显示启动状态
        features = []
        if self.use_vad and self.vad_model is not None:
            features.append("语音端点检测")
        features.append("语音识别")
        if self.use_punc and self.punc_model is not None:
            features.append("标点恢复")

        log(f"语音识别已启动，包含" + "、".join(features) + "功能")
        log("请对着麦克风说话...")
        return True

    def stop(self):
        """停止语音识别"""
        if not self.running:
            return

        self.running = False

        # 停止音频流
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                log(f"停止音频流错误: {e}")

        # 等待线程结束
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)

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
            log(f"最终处理错误: {e}")

        log("语音识别已停止。")
        log(f"最终转写结果: {self.complete_transcript}")


#----------------------------------------
# 简化版翻译模块
#----------------------------------------
class SimpleTranslator:
    """简化版翻译模块"""

    def __init__(self, app_id, api_secret, api_key):
        """初始化翻译模块"""
        log("初始化翻译模块...")
        self.app_id = app_id
        self.api_secret = api_secret
        self.api_key = api_key
        self.url = 'https://itrans.xf-yun.com/v1/its'

    def translate(self, text, from_lang="cn", to_lang="en"):
        """翻译文本"""
        if not text or from_lang == to_lang:
            return text

        try:
            log(f"翻译文本: '{text[:20]}...' 从 {from_lang} 到 {to_lang}")

            # 生成URL和请求头
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            # 解析URL
            url_parts = self.parse_url(self.url)
            host = url_parts["host"]
            path = url_parts["path"]

            # 生成签名
            signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/1.1"
            signature_sha = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_origin.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
            signature_sha = base64.b64encode(signature_sha).decode('utf-8')

            # 构建authorization
            authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

            # 构建URL
            request_url = self.url + "?" + urlencode({
                "host": host,
                "date": date,
                "authorization": authorization
            })

            # 构建请求体
            body = {
                "header": {
                    "app_id": self.app_id,
                    "status": 3
                },
                "parameter": {
                    "its": {
                        "from": from_lang,
                        "to": to_lang,
                        "result": {}
                    }
                },
                "payload": {
                    "input_data": {
                        "encoding": "utf8",
                        "status": 3,
                        "text": base64.b64encode(text.encode("utf-8")).decode('utf-8')
                    }
                }
            }

            # 准备请求头
            headers = {
                'content-type': "application/json",
                'host': host,
                'app_id': self.app_id
            }

            # 发送请求
            response = requests.post(
                request_url,
                data=json.dumps(body),
                headers=headers,
                timeout=5.0
            )

            # 解析响应
            if response.status_code == 200:
                result = json.loads(response.content.decode())

                if 'payload' in result and 'result' in result['payload'] and 'text' in result['payload']['result']:
                    translated_text_base64 = result['payload']['result']['text']
                    translated_text = base64.b64decode(translated_text_base64).decode()

                    try:
                        # 尝试解析JSON响应
                        json_result = json.loads(translated_text)
                        if 'trans_result' in json_result and 'dst' in json_result['trans_result']:
                            return json_result['trans_result']['dst']
                        elif 'dst' in json_result:
                            return json_result['dst']
                        else:
                            return translated_text
                    except:
                        # 不是JSON格式，返回原始文本
                        return translated_text
                else:
                    log(f"翻译API错误: {result}")
            else:
                log(f"翻译请求失败，状态码: {response.status_code}")

            return None

        except Exception as e:
            log(f"翻译过程出错: {str(e)}")
            return None

    def parse_url(self, url):
        """解析URL"""
        stidx = url.index("://")
        host = url[stidx + 3:]
        schema = url[:stidx + 3]
        edidx = host.index("/")
        if edidx <= 0:
            raise Exception("invalid request url:" + url)
        path = host[edidx:]
        host = host[:edidx]
        return {"host": host, "path": path, "schema": schema}


#----------------------------------------
# GUI应用
#----------------------------------------
class ASRTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # 翻译API配置
        self.APP_ID = "86c79fb7"
        self.API_SECRET = "MDY3ZGFkYWEyZDBiOTJkOGIyOTllOWMz"
        self.API_KEY = "f4369644e37eddd43adfe436e7904cf1"

        # 配置主窗口
        self.title("语音识别翻译系统")
        self.geometry("800x600")
        self.minsize(600, 400)

        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))

        # 状态变量
        self.is_recording = False
        self.source_lang = "cn"  # 默认源语言：中文
        self.target_lang = "en"  # 默认目标语言：英语

        # 语言映射
        self.languages = {
            "cn": "中文",
            "en": "英文",
            "ja": "日语",
            "ko": "韩语",
            "fr": "法语",
            "es": "西班牙语",
            "ru": "俄语",
            "de": "德语",
            "it": "意大利语"
        }

        # 初始化组件
        self.create_widgets()

        # 初始化翻译模块
        self.translator = SimpleTranslator(
            app_id=self.APP_ID,
            api_secret=self.API_SECRET,
            api_key=self.API_KEY
        )

        # 初始化语音识别模块（如果库可用）
        if FUNASR_AVAILABLE:
            self.asr = FastLoadASR(use_vad=True, use_punc=True)
            self.update_status("模型加载中，请稍候...")
        else:
            self.asr = None
            self.update_status("语音识别库未安装，仅翻译功能可用")
            self.record_button.config(state="disabled")

        # 启动日志更新线程
        self.update_log_thread = threading.Thread(target=self.update_log, daemon=True)
        self.update_log_thread.start()

        # 启动结果更新线程
        if FUNASR_AVAILABLE:
            self.update_result_thread = threading.Thread(target=self.update_result, daemon=True)
            self.update_result_thread.start()

        # 在关闭窗口时停止所有进程
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 语言选择区域
        lang_frame = ttk.Frame(control_frame)
        lang_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 源语言标签（仅显示）
        ttk.Label(lang_frame, text="源语言:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.source_lang_var = tk.StringVar(value=self.languages[self.source_lang])
        ttk.Label(lang_frame, textvariable=self.source_lang_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # 目标语言下拉菜单
        ttk.Label(lang_frame, text="目标语言:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.target_lang_var = tk.StringVar(value=self.target_lang)

        # 将语言代码映射为显示名称
        lang_options = [f"{code} ({name})" for code, name in self.languages.items() if code != self.source_lang]

        self.target_lang_combo = ttk.Combobox(lang_frame, values=lang_options, state="readonly", width=15)
        self.target_lang_combo.set(f"{self.target_lang} ({self.languages[self.target_lang]})")
        self.target_lang_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.target_lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)

        # 录音按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)

        self.record_button = ttk.Button(button_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.pack(side=tk.RIGHT, padx=5)

        # 状态条
        self.status_var = tk.StringVar(value="系统就绪")
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=5, pady=2)

        # 创建中间部分的Notebook（选项卡）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 创建"结果"选项卡
        results_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(results_frame, text="语音识别结果")

        # 原文区域
        original_frame = ttk.LabelFrame(results_frame, text=f"原文 ({self.languages[self.source_lang]})")
        original_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.original_text = scrolledtext.ScrolledText(original_frame, wrap=tk.WORD, height=5)
        self.original_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 翻译区域
        translation_frame = ttk.LabelFrame(results_frame, text=f"译文 ({self.languages[self.target_lang]})")
        translation_frame.pack(fill=tk.BOTH, expand=True)

        self.translation_text = scrolledtext.ScrolledText(translation_frame, wrap=tk.WORD, height=5)
        self.translation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建"日志"选项卡
        log_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(log_frame, text="系统日志")

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_change(self, event):
        """处理语言选择变化"""
        selection = self.target_lang_combo.get()
        self.target_lang = selection.split()[0]  # 从显示格式中提取语言代码
        ttk.Label(self.notebook.nametowidget(self.notebook.select()).winfo_children()[1],
                 text=f"译文 ({self.languages[self.target_lang]})").config(text=f"译文 ({self.languages[self.target_lang]})")
        self.update_status(f"目标语言已更改为: {self.languages[self.target_lang]}")

    def toggle_recording(self):
        """切换录音状态"""
        if not FUNASR_AVAILABLE:
            self.update_status("错误: 语音识别库未安装")
            return

        if not self.is_recording:
            # 开始录音
            self.start_recording()
        else:
            # 停止录音
            self.stop_recording()

    def start_recording(self):
        """开始录音"""
        # 检查ASR模型是否已加载
        if not self.asr.ensure_asr_model_loaded():
            self.update_status("错误: ASR模型未加载")
            return

        # 清空结果文本
        self.original_text.delete(1.0, tk.END)
        self.translation_text.delete(1.0, tk.END)

        # 更新UI
        self.is_recording = True
        self.record_button.config(text="停止录音")
        self.update_status("正在录音...")

        # 启动ASR
        success = self.asr.start()
        if not success:
            self.is_recording = False
            self.record_button.config(text="开始录音")
            self.update_status("启动语音识别失败")
            return

        # 切换到结果选项卡
        self.notebook.select(0)

    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return

        # 更新UI
        self.is_recording = False
        self.record_button.config(text="开始录音")
        self.update_status("正在处理最终结果...")

        # 停止ASR
        self.asr.stop()

        # 处理最终结果
        final_text = self.asr.complete_transcript
        if final_text:
            # 更新原文
            self.original_text.delete(1.0, tk.END)
            self.original_text.insert(tk.END, final_text)

            # 翻译最终文本
            final_translation = self.translator.translate(
                text=final_text,
                from_lang=self.source_lang,
                to_lang=self.target_lang
            )

            if final_translation:
                # 更新译文
                self.translation_text.delete(1.0, tk.END)
                self.translation_text.insert(tk.END, final_translation)

        self.update_status("录音已停止，处理完成")

    def update_log(self):
        """更新日志显示的线程函数"""
        while True:
            try:
                # 从队列获取日志消息
                while not log_queue.empty():
                    message = log_queue.get_nowait()

                    # 更新日志文本
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)  # 滚动到最新内容

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"更新日志错误: {e}")

    def update_result(self):
        """更新识别和翻译结果的线程函数"""
        while True:
            try:
                if FUNASR_AVAILABLE and hasattr(self, 'asr') and self.asr and self.is_recording:
                    # 检查ASR结果队列
                    while not self.asr.result_queue.empty():
                        _, text = self.asr.result_queue.get_nowait()

                        # 更新原文
                        self.original_text.delete(1.0, tk.END)
                        self.original_text.insert(tk.END, text)

                        # 翻译文本
                        translated = self.translator.translate(
                            text=text,
                            from_lang=self.source_lang,
                            to_lang=self.target_lang
                        )

                        # 更新译文
                        if translated:
                            self.translation_text.delete(1.0, tk.END)
                            self.translation_text.insert(tk.END, translated)

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"更新结果错误: {e}")

    def update_status(self, message):
        """更新状态栏消息"""
        self.status_var.set(message)
        log(message, show_time=False)

    def on_closing(self):
        """窗口关闭处理"""
        if hasattr(self, 'asr') and self.asr and self.asr.running:
            self.asr.stop()
        self.destroy()


# 主函数
def main():
    """主函数"""
    # 打印系统信息
    print("=" * 60)
    print("👂 语音识别翻译系统 - GUI版本")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")

    # 检查是否安装了必要的库
    if not FUNASR_AVAILABLE:
        print("警告: funasr或sounddevice未安装，语音识别功能将无法使用")
        print("请安装必要的库: pip install funasr sounddevice numpy requests")

    # 创建并运行应用
    app = ASRTranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()