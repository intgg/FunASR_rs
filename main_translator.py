# main_translator.py - 实时语音翻译系统主控模块 - 性能优化版
import time
import threading
from queue import Queue
import tkinter as tk
from tkinter import ttk, scrolledtext
import gc
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# 导入优化后的模块
from funasr_module import FunASRModule
from translation_module import TranslationModule
from realtime_tts import RealtimeTTS

# 导入PyAudio用于获取设备列表
import pyaudio


class RealtimeTranslator:
    """实时语音翻译系统 - 性能优化版"""

    # 使用__slots__减少内存占用
    __slots__ = ['p_audio', 'input_devices', 'output_devices',
                 'selected_input_device_index', 'selected_output_device_index',
                 'asr', 'translator', 'tts', 'is_running', 'is_testing',
                 'is_speaking', 'asr_paused', 'auto_pause_recognition',
                 'target_language', 'current_voice', 'voice_speed',
                 'voice_volume', 'supported_languages', 'language_voices',
                 'recognition_queue', 'translation_queue', 'root',
                 'input_device_var', 'output_device_var', 'input_device_combo',
                 'output_device_combo', 'lang_var', 'lang_combo', 'voice_var',
                 'voice_combo', 'start_btn', 'status_label', 'auto_pause_var',
                 'auto_pause_cb', 'auto_pause_hint', 'speed_scale', 'speed_label',
                 'volume_scale', 'volume_label', 'test_btn', 'stop_test_btn',
                 'source_text', 'translated_text', 'thread_pool', 'gui_update_buffer',
                 'gui_update_timer', 'last_gui_update', 'gui_update_interval',
                 'last_gc_time', 'gc_interval', 'performance_stats',
                 'recognition_future', 'translation_future']

    def __init__(self, asr_app_id, asr_api_key, app_id, api_key, api_secret):
        # 创建线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # 初始化PyAudio以获取设备信息
        self.p_audio = pyaudio.PyAudio()

        # 获取设备列表
        self.input_devices = self.get_input_devices()
        self.output_devices = self.get_output_devices()

        # 当前选择的设备索引
        self.selected_input_device_index = None
        self.selected_output_device_index = None

        # 初始化三个模块
        self.asr = FunASRModule(asr_app_id, asr_api_key)
        self.translator = TranslationModule(app_id, api_secret, api_key)
        self.tts = RealtimeTTS(app_id, api_key, api_secret)

        # 状态控制
        self.is_running = False
        self.is_testing = False  # 跟踪是否正在试听
        self.is_speaking = False  # 跟踪TTS是否正在播放
        self.asr_paused = False  # 跟踪ASR是否被暂停
        self.auto_pause_recognition = True  # 是否自动暂停识别的开关
        self.target_language = "en"  # 默认翻译为英文

        # 语音合成设置
        self.current_voice = "x4_enus_luna_assist"  # 默认音色
        self.voice_speed = 50  # 默认语速
        self.voice_volume = 60  # 默认音量

        # 支持的语言和对应的音色
        self.supported_languages = {
            "en": {"name": "英语", "code": "en", "default_voice": "x4_enus_luna_assist"},
            "cn": {"name": "中文", "code": "cn", "default_voice": "x4_gaolengnanshen_talk"},
            "id": {"name": "印尼语", "code": "id", "default_voice": "x2_IdId_Kris"},
            "vi": {"name": "越南语", "code": "vi", "default_voice": "x2_ViVn_ThuHien"},
            "th": {"name": "泰语", "code": "th", "default_voice": "yingying"},
            "ja": {"name": "日语", "code": "ja", "default_voice": "qianhui"},
            "es": {"name": "西班牙语", "code": "es", "default_voice": "x2_spes_aurora"},
            "pt": {"name": "葡萄牙语", "code": "pt", "default_voice": "maria"},
            "it": {"name": "意大利语", "code": "it", "default_voice": "x2_ItIt_Anna"},
            "hi": {"name": "印地语", "code": "hi", "default_voice": "x2_HiIn_Mohita"}
        }

        # 语言到音色的映射
        self.language_voices = {
            "en": ["x4_enus_luna_assist", "x4_enus_gavin_assist"],  # 英语音色
            "cn": ["x4_gaolengnanshen_talk", "x4_panting"],  # 中文音色
            "id": ["x2_IdId_Kris"],  # 印尼语音色
            "vi": ["x2_ViVn_ThuHien"],  # 越南语音色
            "th": ["yingying"],  # 泰语音色
            "ja": ["qianhui", "x4_jajp_zhongcun_assist"],  # 日语音色
            "es": ["x2_spes_aurora"],  # 西班牙语音色
            "pt": ["maria"],  # 葡萄牙语音色
            "it": ["x2_ItIt_Anna"],  # 意大利语音色
            "hi": ["x2_HiIn_Mohita"]  # 印地语音色
        }

        # 结果队列
        self.recognition_queue = Queue()
        self.translation_queue = Queue()

        # 异步任务引用
        self.recognition_future = None
        self.translation_future = None

        # GUI更新缓冲
        self.gui_update_buffer = {"source": deque(maxlen=10), "translation": deque(maxlen=10)}
        self.last_gui_update = time.time()
        self.gui_update_interval = 0.1  # 更新间隔(秒)

        # 垃圾回收控制
        self.last_gc_time = time.time()
        self.gc_interval = 60  # 每分钟执行一次垃圾回收

        # 性能统计
        self.performance_stats = {
            "recognition_time": deque(maxlen=50),
            "translation_time": deque(maxlen=50),
            "tts_time": deque(maxlen=50)
        }

        # 创建GUI
        self.create_gui()

        # 设置更新计时器
        self.setup_timers()

    def get_input_devices(self):
        """获取所有输入设备"""
        input_devices = []

        for i in range(self.p_audio.get_device_count()):
            device_info = self.p_audio.get_device_info_by_index(i)
            # 只添加输入设备（麦克风）
            if device_info['maxInputChannels'] > 0:
                name = device_info['name']
                input_devices.append({'index': i, 'name': name})

        return input_devices

    def get_output_devices(self):
        """获取所有输出设备"""
        output_devices = []

        for i in range(self.p_audio.get_device_count()):
            device_info = self.p_audio.get_device_info_by_index(i)
            # 只添加输出设备（扬声器）
            if device_info['maxOutputChannels'] > 0:
                name = device_info['name']
                output_devices.append({'index': i, 'name': name})

        return output_devices

    def update_input_device(self, event=None):
        """更新输入设备 - 优化设备切换"""
        selected_name = self.input_device_var.get()

        # 找到对应的设备索引
        device_index = None
        for device in self.input_devices:
            if device['name'] == selected_name:
                device_index = device['index']
                break

        if device_index is not None and device_index != self.selected_input_device_index:
            self.selected_input_device_index = device_index
            print(f"输入设备已更改为: {selected_name} (索引: {device_index})")

            # 如果正在运行，使用热切换方式更新
            was_running = self.is_running
            if was_running:
                # 先标记为暂停状态
                self.asr_paused = True
                # 停止当前识别
                self.asr.stop()

                # 等待资源释放
                time.sleep(0.3)

                # 检查ASR模块是否支持热切换设备
                if hasattr(self.asr, 'update_device'):
                    # 使用热切换功能
                    updated = self.asr.update_device(device_index)
                    if updated:
                        self.update_gui("source", f"[已切换输入设备: {selected_name} (热切换)]\n")
                    else:
                        # 如果热切换失败，则重新初始化
                        self.asr = FunASRModule(None, None, input_device_index=device_index)
                        self.update_gui("source", f"[已切换输入设备: {selected_name} (重新初始化)]\n")
                else:
                    # 重新初始化ASR模块，使用新设备
                    self.asr = FunASRModule(None, None, input_device_index=device_index)
                    self.update_gui("source", f"[已切换输入设备: {selected_name}]\n")

                # 如果之前在运行且没有在播放声音，则重新启动识别
                if was_running and not self.is_speaking:
                    self.asr.start()
                    self.asr_paused = False
                    self.update_status_display()
            else:
                # 仅更新索引，不重新初始化
                self.update_gui("source", f"[已设置输入设备: {selected_name}]\n")

    def update_output_device(self, event=None):
        """更新输出设备 - 优化设备切换"""
        selected_name = self.output_device_var.get()

        # 找到对应的设备索引
        device_index = None
        for device in self.output_devices:
            if device['name'] == selected_name:
                device_index = device['index']
                break

        if device_index is not None and device_index != self.selected_output_device_index:
            self.selected_output_device_index = device_index
            print(f"输出设备已更改为: {selected_name} (索引: {device_index})")

            # 如果正在播放，先停止
            was_speaking = self.is_speaking
            was_testing = self.is_testing

            if was_speaking or was_testing:
                self.tts.stop_speaking()
                # 给点时间让资源释放
                time.sleep(0.2)

            # 使用热切换功能
            if hasattr(self.tts, 'update_device'):
                self.tts.update_device(device_index)
                self.update_gui("translation", f"[已切换输出设备: {selected_name} (热切换)]\n")
            else:
                # 重新初始化TTS模块，使用新设备
                self.tts = RealtimeTTS(self.tts.app_id, self.tts.api_key, self.tts.api_secret,
                                       output_device_index=device_index)
                self.update_gui("translation", f"[已切换输出设备: {selected_name}]\n")

    def pause_recognition(self):
        """暂停语音识别"""
        if self.is_running and not self.asr_paused:
            print("暂停语音识别...")
            self.asr.stop()  # 停止WebSocket连接
            self.asr_paused = True
            self.update_gui("source", "[系统正在播放语音，识别已暂停]\n")
            self.update_status_display()  # 更新状态显示

    def resume_recognition(self):
        """恢复语音识别"""
        if self.is_running and self.asr_paused:
            print("恢复语音识别...")
            self.asr.start()  # 重新启动WebSocket连接
            self.asr_paused = False
            self.update_gui("source", "[识别已恢复]\n")
            self.update_status_display()  # 更新状态显示

    def speak_with_coordination(self, text, voice=None, speed=None, volume=None):
        """根据用户设置协调语音播放和识别"""
        # 标记为正在播放状态
        self.is_speaking = True

        # 记录开始时间
        start_time = time.time()

        # 仅在用户启用了自动暂停功能时暂停识别
        if self.auto_pause_recognition:
            self.pause_recognition()
            # 添加短暂延迟，确保识别真正停止
            time.sleep(0.2)
        elif not self.asr_paused:
            # 如果没有启用自动暂停，但识别已被其他原因暂停，保持现状
            # 否则更新状态显示
            self.update_status_display()

        # 设置TTS播放完成的回调
        def on_playback_finished():
            # 计算TTS时间并记录
            tts_time = time.time() - start_time
            self.performance_stats["tts_time"].append(tts_time)

            # 更新状态
            self.is_speaking = False

            # 仅在之前由于播放而暂停识别的情况下恢复识别
            if self.auto_pause_recognition and self.asr_paused:
                # 添加短暂延迟，确保TTS完全停止后再恢复识别
                time.sleep(0.1)
                self.resume_recognition()
            else:
                # 仅更新状态显示
                self.update_status_display()

        # 将回调函数传递给TTS模块
        self.tts.speak(text, voice, speed, volume, callback=on_playback_finished)

    def setup_timers(self):
        """设置定时器"""
        # GUI更新定时器
        self.gui_update_timer = self.root.after(100, self.flush_gui_updates)

        # 设置垃圾回收定时器
        def check_gc():
            current_time = time.time()
            if current_time - self.last_gc_time >= self.gc_interval:
                gc.collect()
                self.last_gc_time = current_time

            # 继续检查
            self.root.after(5000, check_gc)  # 每5秒检查一次

        # 启动垃圾回收定时器
        self.root.after(5000, check_gc)

        # 监控TTS播放状态
        def check_playing_status():
            # 检查TTS模块的播放状态
            if self.is_testing and hasattr(self.tts, 'is_playing') and not self.tts.is_playing:
                # 如果标记为正在测试，但实际上已经不在播放了
                self.is_testing = False
                self.test_btn.config(state=tk.NORMAL)
                self.stop_test_btn.config(state=tk.DISABLED)

            # 每200毫秒检查一次
            self.root.after(200, check_playing_status)

        # 启动播放状态检查
        self.root.after(200, check_playing_status)

    def flush_gui_updates(self):
        """刷新GUI更新缓冲区"""
        current_time = time.time()

        # 检查是否应该更新GUI
        if current_time - self.last_gui_update >= self.gui_update_interval:
            # 处理源文本更新
            if self.gui_update_buffer["source"]:
                for text, append in self.gui_update_buffer["source"]:
                    if append:
                        self.source_text.insert(tk.END, text)
                        self.source_text.see(tk.END)
                    else:
                        # 更新最后一行（用于中间结果）
                        self.source_text.delete("end-2l", tk.END)
                        self.source_text.insert(tk.END, text + "\n")
                        self.source_text.see(tk.END)

                # 清空缓冲区
                self.gui_update_buffer["source"].clear()

            # 处理翻译文本更新
            if self.gui_update_buffer["translation"]:
                for text, append in self.gui_update_buffer["translation"]:
                    self.translated_text.insert(tk.END, text)
                    self.translated_text.see(tk.END)

                # 清空缓冲区
                self.gui_update_buffer["translation"].clear()

            self.last_gui_update = current_time

        # 继续定时检查
        self.gui_update_timer = self.root.after(50, self.flush_gui_updates)

    def update_status_display(self):
        """更新状态显示"""
        if not self.is_running:
            self.status_label.config(text="状态: 待机", foreground="gray")
        elif self.is_speaking and self.asr_paused:
            self.status_label.config(text="状态: 播放中 (识别已暂停)", foreground="orange")
        elif self.is_speaking and not self.asr_paused:
            self.status_label.config(text="状态: 播放中 (识别进行中)", foreground="blue")
        elif self.asr_paused:
            self.status_label.config(text="状态: 识别已暂停", foreground="orange")
        else:
            self.status_label.config(text="状态: 运行中", foreground="green")

    def detect_language(self, text):
        """简单的语言检测"""
        # 检测中文
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return "cn"

        # 这里可以扩展更多语言检测逻辑
        # 暂时默认其他文本为英文
        return "en"

    def create_gui(self):
        """创建图形界面"""
        self.root = tk.Tk()
        self.root.title("多语种实时语音翻译系统 - 性能优化版")
        self.root.geometry("900x700")

        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === 设备选择区域 ===
        device_frame = ttk.LabelFrame(main_frame, text="音频设备设置")
        device_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建设备选择行
        device_control_frame = ttk.Frame(device_frame)
        device_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 输入设备选择
        ttk.Label(device_control_frame, text="输入设备:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.input_device_var = tk.StringVar()

        # 提取设备名称列表
        input_device_names = [device['name'] for device in self.input_devices]

        # 创建下拉菜单
        self.input_device_combo = ttk.Combobox(device_control_frame,
                                               textvariable=self.input_device_var,
                                               values=input_device_names,
                                               state="readonly",
                                               width=30)
        self.input_device_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W + tk.E)

        # 如果有设备，默认选择第一个
        if input_device_names:
            self.input_device_combo.current(0)
            # 获取默认设备索引
            self.selected_input_device_index = self.input_devices[0]['index']

        # 绑定选择事件
        self.input_device_combo.bind('<<ComboboxSelected>>', self.update_input_device)

        # 输出设备选择
        ttk.Label(device_control_frame, text="输出设备:").grid(row=0, column=2, padx=(20, 5), pady=5, sticky=tk.W)
        self.output_device_var = tk.StringVar()

        # 提取设备名称列表
        output_device_names = [device['name'] for device in self.output_devices]

        # 创建下拉菜单
        self.output_device_combo = ttk.Combobox(device_control_frame,
                                                textvariable=self.output_device_var,
                                                values=output_device_names,
                                                state="readonly",
                                                width=30)
        self.output_device_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W + tk.E)

        # 如果有设备，默认选择第一个
        if output_device_names:
            self.output_device_combo.current(0)
            # 获取默认设备索引
            self.selected_output_device_index = self.output_devices[0]['index']

        # 绑定选择事件
        self.output_device_combo.bind('<<ComboboxSelected>>', self.update_output_device)

        # 刷新设备按钮
        refresh_btn = ttk.Button(device_control_frame, text="刷新设备列表",
                                 command=self.refresh_devices)
        refresh_btn.grid(row=0, column=4, padx=10, pady=5)

        # 设置列权重使下拉框可以扩展
        device_control_frame.columnconfigure(1, weight=1)
        device_control_frame.columnconfigure(3, weight=1)

        # 上部分：语言和控制区域
        control_frame = ttk.LabelFrame(main_frame, text="翻译控制")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：语言选择和开始/停止按钮
        lang_control_frame = ttk.Frame(control_frame)
        lang_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 目标语言选择
        ttk.Label(lang_control_frame, text="目标语言:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # 语言下拉菜单
        self.lang_var = tk.StringVar()
        lang_values = [lang["name"] for lang in self.supported_languages.values()]
        self.lang_combo = ttk.Combobox(lang_control_frame, textvariable=self.lang_var,
                                       values=lang_values, state="readonly", width=10)
        self.lang_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.lang_combo.current(0)  # 默认选择第一个语言（英语）

        # 监听语言选择变化
        self.lang_combo.bind('<<ComboboxSelected>>', self.on_language_change)

        # 音色选择
        ttk.Label(lang_control_frame, text="音色:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(lang_control_frame, textvariable=self.voice_var,
                                        state="readonly", width=15)
        self.voice_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W + tk.E)

        # 初始化音色选项（英语音色）
        self.update_voice_options("en")

        # 监听音色变化
        self.voice_combo.bind('<<ComboboxSelected>>', self.on_voice_change)

        # 开始/停止按钮
        self.start_btn = ttk.Button(lang_control_frame, text="开始翻译",
                                    command=self.toggle_translation)
        self.start_btn.grid(row=0, column=4, padx=20, pady=5)

        # 状态显示
        self.status_label = ttk.Label(lang_control_frame, text="状态: 待机", foreground="gray")
        self.status_label.grid(row=0, column=5, padx=5, pady=5)

        # 清空按钮
        clear_btn = ttk.Button(lang_control_frame, text="清空显示", command=self.clear_display)
        clear_btn.grid(row=0, column=6, padx=5, pady=5)

        # 添加播放时暂停识别的选项行
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # 添加播放时暂停识别的复选框
        self.auto_pause_var = tk.BooleanVar(value=True)  # 默认开启
        self.auto_pause_cb = ttk.Checkbutton(
            options_frame,
            text="播放时暂停识别（避免语音干扰）",
            variable=self.auto_pause_var,
            command=self.toggle_auto_pause
        )
        self.auto_pause_cb.pack(side=tk.LEFT, padx=10)

        # 添加提示标签（当禁用时显示）
        self.auto_pause_hint = ttk.Label(
            options_frame,
            text="注意：关闭此选项可能导致系统识别到自己播放的声音",
            foreground="orange"
        )
        # 默认隐藏提示
        if self.auto_pause_var.get():
            self.auto_pause_hint.pack_forget()
        else:
            self.auto_pause_hint.pack(side=tk.LEFT, padx=10)

        # 第二行：语速和音量控制
        voice_control_frame = ttk.Frame(control_frame)
        voice_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 语速控制
        ttk.Label(voice_control_frame, text="语速:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # 语速滑块（1-10对应0-100）
        def update_speed(val):
            user_speed = int(float(val))
            self.voice_speed = int((user_speed - 1) * (100 / 9))
            self.speed_label.config(text=f"{user_speed}/10")

        self.speed_scale = ttk.Scale(voice_control_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                     command=update_speed, value=5, length=200)
        self.speed_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W + tk.E)

        self.speed_label = ttk.Label(voice_control_frame, text="5/10")
        self.speed_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        # 音量控制
        ttk.Label(voice_control_frame, text="音量:").grid(row=0, column=3, padx=15, pady=5, sticky=tk.W)

        # 音量滑块（1-10对应0-100）
        def update_volume(val):
            user_volume = int(float(val))
            self.voice_volume = int((user_volume - 1) * (100 / 9))
            self.volume_label.config(text=f"{user_volume}/10")

        self.volume_scale = ttk.Scale(voice_control_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                      command=update_volume, value=6, length=200)
        self.volume_scale.grid(row=0, column=4, padx=5, pady=5, sticky=tk.W + tk.E)

        self.volume_label = ttk.Label(voice_control_frame, text="6/10")
        self.volume_label.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)

        # 测试和停止按钮
        test_frame = ttk.Frame(voice_control_frame)
        test_frame.grid(row=0, column=6, padx=10, pady=5)

        # 测试按钮
        self.test_btn = ttk.Button(test_frame, text="试听音色",
                                   command=self.test_current_voice)
        self.test_btn.pack(side=tk.LEFT, padx=5)

        # 停止试听按钮
        self.stop_test_btn = ttk.Button(test_frame, text="停止试听",
                                        command=self.stop_test_voice, state=tk.DISABLED)
        self.stop_test_btn.pack(side=tk.LEFT, padx=5)

        # 中间部分：显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # 原文显示
        ttk.Label(display_frame, text="语音识别结果:").pack(anchor=tk.W)
        self.source_text = scrolledtext.ScrolledText(display_frame, height=8)
        self.source_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 译文显示
        ttk.Label(display_frame, text="翻译结果:").pack(anchor=tk.W)
        self.translated_text = scrolledtext.ScrolledText(display_frame, height=8)
        self.translated_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 底部：使用说明
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # 支持的语言提示
        supported_langs = ", ".join([lang["name"] for lang in self.supported_languages.values()])
        ttk.Label(info_frame,
                  text=f"支持的目标语言: {supported_langs}",
                  foreground="blue").pack(pady=2)

        ttk.Label(info_frame,
                  text="系统会自动检测输入语言，并翻译为所选目标语言",
                  foreground="blue").pack(pady=2)

        # 试听提示
        ttk.Label(info_frame,
                  text='试听功能: 点击"试听音色"体验当前设置，可随时点击"停止试听"终止播放。试听过程中界面可以正常操作。',
                  foreground="green").pack(pady=2)

        # 初始调用一次语言变更，设置默认值
        self.on_language_change()

    def refresh_devices(self):
        """刷新音频设备列表"""
        # 重新获取设备列表
        self.input_devices = self.get_input_devices()
        self.output_devices = self.get_output_devices()

        # 更新下拉菜单
        input_device_names = [device['name'] for device in self.input_devices]
        self.input_device_combo['values'] = input_device_names

        output_device_names = [device['name'] for device in self.output_devices]
        self.output_device_combo['values'] = output_device_names

        # 如果有设备，确保选中第一个
        if input_device_names and not self.input_device_var.get() in input_device_names:
            self.input_device_combo.current(0)
            self.selected_input_device_index = self.input_devices[0]['index']

        if output_device_names and not self.output_device_var.get() in output_device_names:
            self.output_device_combo.current(0)
            self.selected_output_device_index = self.output_devices[0]['index']

        # 提示用户
        self.update_gui("source", "[设备列表已刷新]\n")

    def toggle_auto_pause(self):
        """处理自动暂停识别选项的变化"""
        self.auto_pause_recognition = self.auto_pause_var.get()

        # 更新界面提示
        if self.auto_pause_recognition:
            self.auto_pause_hint.pack_forget()
        else:
            self.auto_pause_hint.pack(side=tk.LEFT, padx=10)

        print(f"{'启用' if self.auto_pause_recognition else '禁用'}播放时暂停识别功能")

        # 如果当前正在播放并且改为不暂停识别，则恢复识别
        if self.is_speaking and not self.auto_pause_recognition and self.asr_paused:
            self.resume_recognition()

    def on_language_change(self, event=None):
        """处理语言变更"""
        # 获取选择的语言名称
        selected_lang_name = self.lang_var.get()

        # 查找对应的语言代码
        selected_lang_code = None
        for code, info in self.supported_languages.items():
            if info["name"] == selected_lang_name:
                selected_lang_code = code
                break

        if selected_lang_code:
            # 更新目标语言
            self.target_language = selected_lang_code

            # 更新可用的音色选项
            self.update_voice_options(selected_lang_code)

    def update_voice_options(self, lang_code):
        """根据语言更新音色选项"""
        # 获取该语言支持的音色
        voices = self.language_voices.get(lang_code, [])

        # 获取音色名称
        voice_names = []
        for voice_id in voices:
            name = self.tts.available_voices.get(voice_id, voice_id)
            voice_names.append(f"{name} ({voice_id})")

        # 更新下拉菜单
        self.voice_combo['values'] = voice_names

        # 选择默认音色
        default_voice = self.supported_languages[lang_code]["default_voice"]
        self.current_voice = default_voice

        # 找到默认音色的索引
        default_index = 0
        for i, voice_id in enumerate(voices):
            if voice_id == default_voice:
                default_index = i
                break

        # 设置当前选中值
        if voice_names:
            self.voice_combo.current(default_index)

    def on_voice_change(self, event=None):
        """处理音色变更"""
        selected_voice = self.voice_var.get()

        # 从选中值中提取音色ID（假设格式为"名称 (ID)"）
        if selected_voice:
            voice_id = selected_voice.split("(")[-1].strip(")")
            self.current_voice = voice_id

    def test_current_voice(self):
        """测试当前选择的音色（非阻塞）"""
        # 如果正在试听，先停止
        if self.is_testing:
            self.stop_test_voice()

        # 根据当前语言获取测试文本
        test_texts = {
            "cn": '这是一段中文测试，当前语速和音量设置。你可以随时点击"停止试听"按钮来中断播放。试听过程中，界面可以正常操作。',
            "en": "This is an English test for the current speed and volume settings. You can click the 'Stop Testing' button at any time to interrupt playback. The interface remains fully operational during playback.",
            "id": "Ini adalah tes bahasa Indonesia untuk pengaturan kecepatan dan volume saat ini. Anda dapat mengklik tombol 'Berhenti Mendengarkan' kapan saja untuk menghentikan pemutaran.",
            "vi": "Đây là bài kiểm tra tiếng Việt cho cài đặt tốc độ và âm lượng hiện tại. Bạn có thể nhấp vào nút 'Dừng Thử' bất kỳ lúc nào để ngắt phát lại.",
            "th": "นี่คือการทดสอบภาษาไทยสำหรับการตั้งค่าความเร็วและระดับเสียงปัจจุบัน คุณสามารถคลิกปุ่ม 'หยุดการทดสอบ' เมื่อใดก็ได้เพื่อหยุดการเล่น",
            "ja": "これは現在の速度と音量設定のための日本語テストです。いつでも「試聴停止」ボタンをクリックすると、再生を中断できます。再生中もインターフェースは完全に操作可能です。",
            "es": "Esta es una prueba en español para la configuración actual de velocidad y volumen. Puede hacer clic en el botón 'Detener prueba' en cualquier momento para interrumpir la reproducción.",
            "pt": "Este é um teste em português para as configurações atuais de velocidade e volume. Você pode clicar no botão 'Parar Teste' a qualquer momento para interromper a reprodução.",
            "it": "Questo è un test in italiano per le impostazioni correnti di velocità e volume. È possibile fare clic sul pulsante 'Interrompi test' in qualsiasi momento per interrompere la riproduzione.",
            "hi": "यह वर्तमान गति और वॉल्यूम सेटिंग्स के लिए हिंदी में एक परीक्षण है। आप किसी भी समय 'परीक्षण रोकें' बटन पर क्लिक कर सकते हैं।"
        }

        # 获取当前语言的测试文本，如果没有则使用英文
        test_text = test_texts.get(self.target_language, test_texts["en"])

        # 标记为正在试听
        self.is_testing = True

        # 更新按钮状态
        self.test_btn.config(state=tk.DISABLED)
        self.stop_test_btn.config(state=tk.NORMAL)

        # 非阻塞方式播放测试文本
        self.tts.speak(test_text, voice=self.current_voice,
                       speed=self.voice_speed, volume=self.voice_volume)

    def stop_test_voice(self):
        """停止试听音色"""
        if self.is_testing:
            # 调用TTS的停止方法
            self.tts.stop_speaking()

            # 更新状态和按钮
            self.is_testing = False
            self.test_btn.config(state=tk.NORMAL)
            self.stop_test_btn.config(state=tk.DISABLED)

    def toggle_translation(self):
        """开始/停止翻译"""
        if not self.is_running:
            self.start_translation()
        else:
            self.stop_translation()

    def start_translation(self):
        """开始翻译"""
        # 如果正在试听，先停止
        if self.is_testing:
            self.stop_test_voice()

        self.is_running = True
        self.start_btn.config(text="停止翻译")
        self.status_label.config(text="状态: 运行中", foreground="green")

        # 启动语音识别
        self.asr.start()

        # 使用线程池启动处理线程
        self.recognition_future = self.thread_pool.submit(self.process_recognition)
        self.translation_future = self.thread_pool.submit(self.process_translation)

        # 更新状态显示
        self.update_status_display()

    def stop_translation(self):
        """停止翻译"""
        self.is_running = False
        self.start_btn.config(text="开始翻译")
        self.status_label.config(text="状态: 待机", foreground="gray")

        # 停止语音识别
        self.asr.stop()

        # 清空播放队列
        self.tts.clear_queue()

        # 更新状态显示
        self.update_status_display()

    def clear_display(self):
        """清空显示区域"""
        self.source_text.delete(1.0, tk.END)
        self.translated_text.delete(1.0, tk.END)

    def process_recognition(self):
        """处理语音识别结果"""
        intermediate_text = ""

        while self.is_running:
            try:
                result = self.asr.get_result()
                if result:
                    if isinstance(result, dict):
                        if "error" in result:
                            self.update_gui("source", f"错误: {result['error']}")
                            self.stop_translation()
                            break
                        elif "text" in result:
                            if result["is_final"]:
                                # 最终结果
                                text = result["text"]
                                # 如果有中间结果，先清除它
                                if intermediate_text:
                                    self.update_gui("source", "", append=False)
                                    intermediate_text = ""
                                self.update_gui("source", text + "\n")
                                # 将最终结果放入翻译队列
                                if text.strip():
                                    self.recognition_queue.put(text)
                            else:
                                # 中间结果 - 显示在同一行
                                text = result["text"]
                                if text != intermediate_text:
                                    intermediate_text = text
                                    self.update_gui("source", f"[识别中] {text}", append=False)
            except Exception as e:
                print(f"识别处理错误: {e}")
                # 添加短暂暂停，减少CPU使用
                time.sleep(0.05)
                continue

            # 添加短暂暂停，减少CPU使用
            time.sleep(0.1)

    def process_translation(self):
        """处理翻译和语音合成"""
        while self.is_running:
            try:
                if not self.recognition_queue.empty() and not self.is_speaking:
                    # 获取识别文本
                    text = self.recognition_queue.get()

                    # 记录开始时间
                    translate_start = time.time()

                    # 自动检测源语言
                    source_lang = self.detect_language(text)

                    # 如果源语言与目标语言相同，直接使用原文
                    if source_lang == self.target_language:
                        translated = text
                        # 直接显示在翻译结果区域
                        self.update_gui("translation", translated + "\n")
                        # 使用协调方法朗读原文
                        self.speak_with_coordination(translated, voice=self.current_voice,
                                                     speed=self.voice_speed, volume=self.voice_volume)
                    else:
                        # 执行翻译
                        translated = self.translator.translate(text, source_lang, self.target_language)

                        # 记录翻译时间
                        translate_time = time.time() - translate_start
                        self.performance_stats["translation_time"].append(translate_time)

                        if translated:
                            self.update_gui("translation", translated + "\n")
                            # 使用协调方法朗读翻译结果
                            self.speak_with_coordination(translated, voice=self.current_voice,
                                                         speed=self.voice_speed, volume=self.voice_volume)
                        else:
                            self.update_gui("translation", "[翻译失败]\n")
            except Exception as e:
                print(f"翻译处理错误: {e}")

            # 添加短暂暂停，减少CPU使用
            time.sleep(0.1)

    def update_gui(self, widget_type, text, append=True):
        """更新GUI显示 - 优化版本使用缓冲区"""
        # 将更新添加到缓冲区
        self.gui_update_buffer[widget_type].append((text, append))

    def run(self):
        """运行主程序"""
        try:
            print("启动实时翻译系统...")
            self.root.mainloop()
        except Exception as e:
            print(f"主程序运行错误: {e}")
        finally:
            # 确保清理资源
            print("清理资源...")

            # 停止翻译
            if self.is_running:
                self.stop_translation()

            # 停止播放
            if self.is_testing or self.is_speaking:
                self.tts.stop_speaking()

            # 取消定时器
            if hasattr(self, 'gui_update_timer'):
                self.root.after_cancel(self.gui_update_timer)

            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)

            # 清理PyAudio
            if hasattr(self, 'p_audio'):
                self.p_audio.terminate()

            # 清理模块资源
            if hasattr(self, 'asr'):
                self.asr.cleanup()

            if hasattr(self, 'tts') and hasattr(self.tts, '__del__'):
                self.tts.__del__()

            print("资源清理完成")


if __name__ == "__main__":
    # 不同API服务的密钥配置

    # 机器翻译和语音合成使用的密钥
    APP_ID = "86c79fb7"
    API_KEY = "f4369644e37eddd43adfe436e7904cf1"
    API_SECRET = "MDY3ZGFkYWEyZDBiOTJkOGIyOTllOWMz"

    # 实时语音转写使用的密钥
    ASR_APP_ID = "86c79fb7"
    ASR_API_KEY = "acf74303ddb1af7196de01aadd232feb"

    # 创建并运行翻译系统
    translator = RealtimeTranslator(ASR_APP_ID, ASR_API_KEY, APP_ID, API_KEY, API_SECRET)
    translator.run()