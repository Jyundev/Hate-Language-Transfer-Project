import os, sys, time
# 실행하는 파일이 script(.py) 인지 main.exe(응용프로그램)인지에 따라 루트 디렉토리 변경을 위함.
if getattr(sys, 'frozen', False):
    # main.py로 실행하는 경우
    print(f"sys._MEIPASS : {sys._MEIPASS}")
    program_dir = os.path.dirname(os.path.dirname(os.path.dirname(sys._MEIPASS)))
else:
    # main.exe로 실행하는 경우
    program_dir = (os.path.dirname(os.path.abspath(__file__)))
os.chdir(program_dir)

print(f"Now Directory => {os.getcwd()}")
LSTM_MODEL_PATH = f"{os.getcwd()}/models/lstm_model"
FAST_TEXT_PATH = f"{os.getcwd()}/models/fast_text/fasttext_model.bin"

def get_lstm_list():
    return sorted([file for file in os.listdir(LSTM_MODEL_PATH) if file.endswith('.h5')],reverse=True)

DATA_PATH = f"{os.getcwd()}/data"
STOP_WORD_PATH = f"{DATA_PATH}/stop_word.txt"
BAD_WORD_PATH = f"{DATA_PATH}/bad_word.txt"

def get_stop_word():
    with open(STOP_WORD_PATH, "r",encoding="utf-8") as f:
      return {w.strip() for w in f.readlines()}

def get_bad_word():
    with open(BAD_WORD_PATH,'r',encoding="utf-8") as f:
      return {st.strip() for st in f.readlines()}
    
MAIN_ICO = f"{DATA_PATH}/favicon.ico"