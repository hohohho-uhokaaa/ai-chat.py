import os
import sys

# --- [重要] 環境変数の設定を最初に行う ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 権限エラーを防ぐため絶対パスで固定
os.environ["HF_HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

# --- ライブラリのインポート ---
from datetime import datetime
import torch
import markdown
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QTextEdit, QLabel, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from transformers import AutoModelForCausalLM, AutoTokenizer

# 選択可能なモデル
MODELS = {
    "Qwen 1.5B (軽量・高速)": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen 7B (高性能・要VRAM)": "Qwen/Qwen2.5-Coder-7B-Instruct"
}

# --- AI処理スレッド ---
class AIWorker(QObject):
    finished_loading = pyqtSignal(str)
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None

    def load_model(self, model_id):
        try:
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.finished_loading.emit(model_id.split("/")[-1])
        except Exception as e:
            self.error_occurred.emit(f"ロード失敗: {str(e)}")

    def generate_response(self, user_input):
        try:
            messages = [
                {"role": "system", "content": "あなたは優秀なエンジニアです。Markdown形式で回答してください。"},
                {"role": "user", "content": user_input},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
            
            raw_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            html_content = markdown.markdown(raw_text, extensions=['fenced_code', 'codehilite'])
            self.response_ready.emit(html_content)
        except Exception as e:
            self.error_occurred.emit(f"推論エラー: {str(e)}")

# --- メインウィンドウ ---
class ChatApp(QWidget):
    request_load = pyqtSignal(str)
    request_inference = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_ai_thread()
        self.change_model() # 初期ロード

    def initUI(self):
        self.setWindowTitle('Qwen2.5-Coder Pro (Markdown対応)')
        self.resize(900, 700)
        
        # メイン垂直レイアウト
        self.main_layout = QVBoxLayout()

        # --- 上部ツールバー ---
        top_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS.keys())
        top_layout.addWidget(QLabel("モデル:"))
        top_layout.addWidget(self.model_combo)

        self.load_button = QPushButton("切り替え")
        self.load_button.clicked.connect(self.change_model)
        top_layout.addWidget(self.load_button)

        top_layout.addStretch()

        self.clear_button = QPushButton("クリア")
        self.clear_button.clicked.connect(self.clear_chat)
        top_layout.addWidget(self.clear_button)
        
        self.main_layout.addLayout(top_layout)

        # --- チャット表示エリア ---
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        # CSSでコードブロックを見やすく
        self.output_area.setHtml("<style>pre { background-color: #2b2b2b; color: #f8f8f2; padding: 10px; border-radius: 5px; font-family: 'Courier New'; } </style><i>システム準備中...</i>")
        self.main_layout.addWidget(self.output_area, stretch=1) # stretch=1でここが一番広くなる

        # --- 入力エリア (ここが消えていた可能性があります) ---
        input_container = QVBoxLayout()
        input_container.addWidget(QLabel("プロンプトを入力:"))
        
        bottom_layout = QHBoxLayout()
        self.input_textbox = QLineEdit()
        self.input_textbox.setPlaceholderText("ここに質問を入力してEnter...")
        self.input_textbox.setMinimumHeight(40) # 高さを確保
        self.input_textbox.returnPressed.connect(self.send_message)
        bottom_layout.addWidget(self.input_textbox)

        self.send_button = QPushButton("送信")
        self.send_button.setMinimumHeight(40)
        self.send_button.clicked.connect(self.send_message)
        bottom_layout.addWidget(self.send_button)
        
        input_container.addLayout(bottom_layout)
        self.main_layout.addLayout(input_container)

        self.setLayout(self.main_layout)

    def setup_ai_thread(self):
        self.ai_thread = QThread()
        self.worker = AIWorker()
        self.worker.moveToThread(self.ai_thread)
        self.worker.finished_loading.connect(self.on_model_loaded)
        self.worker.response_ready.connect(self.on_response_received)
        self.worker.error_occurred.connect(self.on_error)
        self.request_load.connect(self.worker.load_model)
        self.request_inference.connect(self.worker.generate_response)
        self.ai_thread.start()

    def change_model(self):
        model_id = MODELS[self.model_combo.currentText()]
        self.output_area.append(f"<hr><b>[システム]:</b> モデルロード中...")
        self.send_button.setEnabled(False)
        self.request_load.emit(model_id)

    def on_model_loaded(self, name):
        self.output_area.append(f"<b>[システム]:</b> {name} が使用可能です。")
        self.send_button.setEnabled(True)

    def clear_chat(self):
        self.output_area.clear()

    def send_message(self):
        text = self.input_textbox.text().strip()
        if text and self.send_button.isEnabled():
            ts = datetime.now().strftime("%H:%M:%S")
            self.output_area.append(f'<div style="color: gray;">[{ts}]</div><b>あなた:</b> {text}')
            self.request_inference.emit(text)
            self.input_textbox.clear()
            self.send_button.setEnabled(False)

    def on_response_received(self, html):
        ts = datetime.now().strftime("%H:%M:%S")
        self.output_area.append(f'<div style="color: gray;">[{ts}]</div><b>AI:</b>')
        self.output_area.insertHtml(html)
        self.output_area.append("<br>")
        self.send_button.setEnabled(True)
        # スクロールを末尾へ
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

    def on_error(self, msg):
        self.output_area.append(f'<p style="color: red;">エラー: {msg}</p>')
        self.send_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())