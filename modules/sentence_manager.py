# modules/sentence_manager.py - 智能句子管理模块

import re
import time
import difflib


class SentenceManager:
    """智能句子管理器 - 处理文本分割、状态跟踪和TTS触发决策"""

    def __init__(self):
        """初始化句子管理器"""
        # 句子状态字典 - 键为句子文本，值为状态信息
        self.sentences = {}

        # 配置参数
        self.stability_threshold = 0.8  # 稳定性阈值(秒)
        self.max_wait_time = 3.0  # 最大等待时间(秒)
        self.min_diff_ratio = 0.2  # 最小差异比例(播放新句子的阈值)

        # 跟踪变量
        self.played_sentences = set()  # 已播放句子集合
        self.last_played_time = 0  # 上次播放时间
        self.last_update_time = 0  # 上次更新时间

    def process_text(self, text, vad_pause_detected=False):
        """处理输入文本，更新句子状态

        参数:
            text: 完整文本 (ASR结果)
            vad_pause_detected: 是否检测到VAD停顿

        返回:
            dict: {
                'sentences_to_translate': 需要翻译的句子列表,
                'sentences_to_play': 可以播放的句子列表
            }
        """
        current_time = time.time()
        self.last_update_time = current_time

        # 分割文本获取句子
        sentences = self._split_into_sentences(text)

        # 更新句子状态
        for sentence in sentences:
            # 清理句子文本
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果是新句子，添加到状态跟踪
            if sentence not in self.sentences:
                self.sentences[sentence] = {
                    'created_at': current_time,
                    'updated_at': current_time,
                    'completion_time': None,  # 完整性确认时间
                    'stability_time': None,  # 稳定性确认时间
                    'is_complete': self._is_complete_sentence(sentence),
                    'is_stable': False,
                    'translated': False,
                    'played': False,
                    'translation': None
                }

                # 检查新句子是否已经完整
                if self.sentences[sentence]['is_complete'] or vad_pause_detected:
                    self.sentences[sentence]['completion_time'] = current_time
            else:
                # 更新已有句子状态
                state = self.sentences[sentence]
                state['updated_at'] = current_time

                # 检查完整性状态是否变化
                is_complete = self._is_complete_sentence(sentence)
                if is_complete and not state['is_complete']:
                    state['is_complete'] = True
                    state['completion_time'] = current_time
                elif vad_pause_detected and not state['is_complete']:
                    state['is_complete'] = True
                    state['completion_time'] = current_time

        # 更新所有句子的状态
        self._update_sentence_states(current_time, vad_pause_detected)

        # 获取需要翻译和播放的句子
        sentences_to_translate = self._get_sentences_to_translate()
        sentences_to_play = self._get_sentences_to_play()

        return {
            'sentences_to_translate': sentences_to_translate,
            'sentences_to_play': sentences_to_play
        }

    def _update_sentence_states(self, current_time, vad_pause_detected):
        """更新所有句子的状态

        参数:
            current_time: 当前时间戳
            vad_pause_detected: 是否检测到VAD停顿
        """
        for sentence, state in list(self.sentences.items()):
            # 已播放的句子跳过处理
            if state['played']:
                continue

            # 检查句子是否稳定 (1. 已标记完整且经过稳定时间, 或 2. 达到最大等待时间)
            if not state['is_stable']:
                time_since_completion = (current_time - state['completion_time']) if state['completion_time'] else 0
                time_since_creation = current_time - state['created_at']

                if (state['is_complete'] and time_since_completion >= self.stability_threshold) or \
                        (time_since_creation >= self.max_wait_time):
                    state['is_stable'] = True
                    state['stability_time'] = current_time
                    print(f"句子稳定: {sentence[:30]}...")

    def _get_sentences_to_translate(self):
        """获取需要翻译的句子列表

        返回:
            list: 需要翻译的句子列表
        """
        sentences_to_translate = []

        for sentence, state in self.sentences.items():
            # 只翻译稳定且未翻译的句子
            if state['is_stable'] and not state['translated'] and not state['played']:
                sentences_to_translate.append(sentence)

        return sentences_to_translate

    def _get_sentences_to_play(self):
        """获取可以播放的句子列表

        返回:
            list: 可以播放的句子列表
        """
        sentences_to_play = []
        current_time = time.time()

        for sentence, state in self.sentences.items():
            # 只播放已翻译但未播放的句子
            if state['is_stable'] and state['translated'] and not state['played']:
                # 检查是否是已播放句子的微小扩展
                if not self._is_minor_extension(sentence):
                    sentences_to_play.append({
                        'text': sentence,
                        'translation': state['translation'],
                        'priority': self._calculate_priority(sentence, state, current_time)
                    })

        # 按优先级排序
        if sentences_to_play:
            sentences_to_play.sort(key=lambda x: x['priority'], reverse=True)

        return sentences_to_play

    def update_translation(self, sentence, translation):
        """更新句子的翻译结果

        参数:
            sentence: 句子文本
            translation: 翻译结果
        """
        if sentence in self.sentences:
            self.sentences[sentence]['translated'] = True
            self.sentences[sentence]['translation'] = translation
            print(f"翻译完成: {sentence[:30]}... -> {translation[:30]}...")

    def mark_as_played(self, sentence):
        """标记句子为已播放

        参数:
            sentence: 句子文本
        """
        if sentence in self.sentences:
            self.sentences[sentence]['played'] = True
            self.played_sentences.add(sentence)
            self.last_played_time = time.time()
            print(f"标记为已播放: {sentence[:30]}...")

    def _split_into_sentences(self, text):
        """将文本分割为句子

        参数:
            text: 完整文本

        返回:
            list: 句子列表
        """
        if not text:
            return []

        # 句子分隔符（中英文标点符号）
        sentence_delimiters = r'(?<=[.。!！?？;；])\s*'

        # 分割文本获取句子列表
        parts = re.split(sentence_delimiters, text)
        sentences = []

        for part in parts:
            part = part.strip()
            if part:
                # 确保句子有结束标点
                if not re.search(r'[.。!！?？;；]\s*$', part):
                    # 暂时不添加标点，保持原样
                    pass
                sentences.append(part)

        return sentences

    def _is_complete_sentence(self, sentence):
        """判断句子是否完整（以标点符号结尾）

        参数:
            sentence: 句子文本

        返回:
            bool: 是否完整
        """
        return bool(re.search(r'[.。!！?？;；]\s*$', sentence))

    def _is_minor_extension(self, sentence):
        """检查是否是已播放句子的微小扩展

        参数:
            sentence: 句子文本

        返回:
            bool: 是否为微小扩展
        """
        # 检查句子是否与已播放句子有很高的相似度
        for played in self.played_sentences:
            # 计算相似度比例
            similarity = difflib.SequenceMatcher(None, played, sentence).ratio()

            # 如果新句子是已播放句子的扩展
            if sentence.startswith(played):
                # 计算新增部分的比例
                diff_ratio = (len(sentence) - len(played)) / len(played)

                # 如果新增部分很小，认为是微小扩展
                if diff_ratio < self.min_diff_ratio:
                    print(f"微小扩展 (忽略): {played[:20]}... -> {sentence[:20]}... (差异比例: {diff_ratio:.2f})")
                    return True

            # 如果两句话非常相似
            elif similarity > 0.9:
                print(f"高度相似 (忽略): {played[:20]}... <-> {sentence[:20]}... (相似度: {similarity:.2f})")
                return True

        return False

    def _calculate_priority(self, sentence, state, current_time):
        """计算句子的播放优先级

        参数:
            sentence: 句子文本
            state: 句子状态
            current_time: 当前时间

        返回:
            float: 优先级分数
        """
        # 基础优先级分数
        priority = 0

        # 1. 完整句子优先
        if state['is_complete']:
            priority += 100

        # 2. 等待时间越长优先级越高
        wait_time = current_time - state['created_at']
        priority += min(wait_time * 10, 50)  # 最多加50分

        # 3. 长句子优先级稍低 (防止长句占用太多时间)
        length_factor = len(sentence) / 50.0  # 标准化长度
        priority -= max(0, length_factor - 1) * 10  # 超过50个字符的部分减分

        return priority

    def clear(self):
        """清空所有状态"""
        self.sentences = {}
        self.played_sentences = set()
        self.last_played_time = 0
        self.last_update_time = 0
        print("句子管理器已重置")