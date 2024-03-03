from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5.QtWidgets import QMainWindow,QTabBar
from interface import Ui_MainWindow
import config as cf
from model import lsmtModel,fastTextModel,textProcessing
import numpy as np
from collections import defaultdict
import time 

class AnalysisThread(QThread):
    change_val = pyqtSignal(int)
    change_stat = pyqtSignal(str)
    # succes_stat = pyqtSignal(bool)

    def __init__(self):
        QThread.__init__(self)
        self.ui = None
        self.text_proc = None
        self.lsmt_model = None
        self.fasttext_model = None
        self.target_text = None
        self.working=False
    
    def set_data(self,ui,text_proc,lsmt_model,fasttext_model):
        self.ui = ui
        self.text_proc = text_proc
        self.lsmt_model = lsmt_model
        self.fasttext_model = fasttext_model

    def set_target(self,target_text):
        self.target_text = target_text

    def run(self):
        self.working=True
        if self.working:
            self.change_val.emit(20)
            self.change_stat.emit("불용어 제거 및 형태소 분리 작업 중...")
            self.ui.plainTextEdit.insertPlainText(f"입력된 문장 => {self.target_text}\n")
            # 1.한글·공백 이외 문자 제거 후, 형태소 분리 및 불용어 제거 수행
            test_word = self.text_proc.tokenize_remove_stop_word(self.text_proc.remove_other_letter(self.target_text))
            time.sleep(2)
            self.ui.plainTextEdit.insertPlainText(f"한글·공백 이외 문자 제거 후 문장  => {test_word}\n")
            # 2.형태소 별 초성·중성·종성으로 분리
            test_word_split = [self.text_proc.separate(word) for word in test_word]
            time.sleep(2)
            self.ui.plainTextEdit.insertPlainText(f"자모단위 분리 후 문장 => {test_word_split}\n")
            self.change_val.emit(40)
            self.change_stat.emit("형태소 데이터를 3차원 ndarray로 변환 중...")
            # 3. 분리된 형태소를 3차원 ndarray 벡터로 변환
            test_word_vec = list()
            for i in range(24):
                if i < len(test_word_split):
                    test_word_vec.append(self.fasttext_model.model[test_word_split[i]])
                else:
                    test_word_vec.append(np.array([0]*100))
            test_word_vec = np.array(test_word_vec)
            test_word_vec = test_word_vec.reshape(1,test_word_vec.shape[0],test_word_vec.shape[1])

            result_pred = self.lsmt_model.predict(test_word_vec)
            time.sleep(2)
            cursor = self.ui.plainTextEdit.textCursor()
            cursor.insertText(f"\n\n{'-'*50}\n\n")
            cursor.insertText("비속어가 포함되어 있을 확률 : ")
            format = QtGui.QTextCharFormat() 
            if result_pred >= 0.5:
                format.setForeground(QtGui.QColor("red"))
            cursor.insertText(str(result_pred), format)
            format.setForeground(QtGui.QColor("black"))
            # cursor.insertText(str(result_pred), QtGui.QTextCharFormat().setBackground(QtGui.QColor("red")))
            cursor.insertText("\n\n",format)
            # self.ui.plainTextEdit.insertPlainText(f"\n\n{'-'*50}\n\n비속어가 포함되어 있을 확률 : <b><font color='red'>{result_pred}</font></b>\n\n")
            
            if result_pred >= 0.5:   
                result_list = list()

                self.change_stat.emit("연관 단어 조회 탐색 중...")
                for idx,word in enumerate(test_word_split):
                    self.change_val.emit(40 + (60//len(test_word_split))*(idx+1))
                    result = self.fasttext_model.model.get_nearest_neighbors(word)
                    similar_letter_dict = defaultdict(list)
                    for _,temp in result:
                        if not self.text_proc.remove_other_letter(temp):
                            continue
                        for w in self.text_proc.bad_word:
                            if w in self.text_proc.separate_reversed(temp):
                                similar_letter_dict[self.text_proc.separate_reversed(word)].append(w)
                    for key,val in similar_letter_dict.items():
                        result_list.append(f"{key} Similarity Texts ==>  {val}")
                        # result_list.append(f"{key} ==> 연관 비속어 : {val}")

                for result in result_list:
                    self.ui.plainTextEdit.insertPlainText(f"{result}\n")
            
            self.change_stat.emit("언어 순화 분석 완료")
            self.change_val.emit(100)
            self.working = False

class TextEdit(QtWidgets.QLineEdit):
    sendTextSignal = QtCore.pyqtSignal(str)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return):
            self.sendTextSignal.emit("Enter key pressed")
        super().keyPressEvent(event)
    

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # interface ui 클래스
        self.ui = Ui_MainWindow()
        #UI 설정
        self.ui.setupUi(self)
        
        # tab bar 숨김 설정
        self.ui.tabWidget.findChild(QTabBar).hide()

        # lstm 모델 저장 변수
        self.lsmt_model = lsmtModel()
        # fasttext 모델 저장 변수
        self.fasttext_model = fastTextModel()
        # 텍스트 관련 전처리 함수
        self.text_pre = textProcessing()
        # 분석 수행 쓰레드 클래스 생성
        self.analysis_thread = AnalysisThread()
        self.analysis_thread.set_data(self.ui
                                      ,self.text_pre
                                      ,self.lsmt_model
                                      ,self.fasttext_model
                                      )
        # 실행 시 최초 모델 로드 함수
        self._model_names_load()

        # 테스트할 문장 입력 관련 위젯 생성
        self.question = TextEdit()
        self.question.setMinimumSize(QtCore.QSize(0, 40))
        self.question.setMaximumSize(QtCore.QSize(16777215, 40))
        self.ui.horizontalLayout_5.addWidget(self.question)

        # 테스트할 문장 결과 전송을 위한 버튼 생성
        self.submitBtn = QtWidgets.QPushButton(self.ui.questionFrame)
        self.submitBtn.setMinimumSize(QtCore.QSize(0, 40))
        self.submitBtn.setMaximumSize(QtCore.QSize(16777215, 40))
        self.submitBtn.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/blueIcons/blueIcons/navigation.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.submitBtn.setIcon(icon3)
        self.submitBtn.setIconSize(QtCore.QSize(32, 32))
        self.submitBtn.setObjectName("submitBtn")
        self.ui.horizontalLayout_5.addWidget(self.submitBtn)

        # 버튼 클릭 시 분석 수행
        self.submitBtn.clicked.connect(self._analysis_text)
        # 테스트할 문장을 버튼 대신 Enter을 입력하였을때, 분석수행
        self.question.sendTextSignal.connect(self._analysis_text)
        

    def _analysis_text(self):
        self.ui.plainTextEdit.clear()
        self.analysis_thread.set_target(self.question.text())

        self.analysis_thread.change_val.connect(self.ui.progressBar.setValue)
        self.analysis_thread.change_stat.connect(self.ui.progressBar.setFormat)

        self.analysis_thread.start()
    


    def _model_names_load(self):
        model_list = cf.get_lstm_list()
        self.ui.modelComboBox.addItems(model_list)
        self.ui.modelComboBox.setCurrentIndex(0)
        self._model_sel_btn_click()
        self.ui.modelComboBox.currentIndexChanged.connect(self._model_sel_btn_click)

    def _model_sel_btn_click(self):
        sel_model = f"{cf.LSTM_MODEL_PATH}/{self.ui.modelComboBox.currentText()}" 
        self.lsmt_model.load_model(sel_model)

if __name__ == "__main__":
    import sys
    def _my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(cf.MAIN_ICO))
    fontDB = QFontDatabase()
    fontDB.addApplicationFont("./NanumBarunGothic.ttf")
    app.setFont(QFont('NanumBarunGothic',10))
    ui = MainWindow()
    ui.show()
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook

    # Set the exception hook to our wrapping function
    sys.excepthook = _my_exception_hook
    sys.exit(app.exec_())
