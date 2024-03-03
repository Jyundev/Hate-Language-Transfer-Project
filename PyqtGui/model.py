import config as cf
from tensorflow import keras
import fasttext
import re
from konlpy.tag import Okt
import numpy as np
from collections import defaultdict

class lsmtModel():
    def __init__(self):
        self.model = None
    
    def load_model(self,model_path):
        self.model = keras.models.load_model(f"{model_path}")
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions[0][0]
        

    
class fastTextModel():
    def __init__(self):
        self.model = fasttext.load_model(cf.FAST_TEXT_PATH)

    # 유사 비속어 개수 확인 함수
    def find_slang(self,test_word_split,text_proc):
        result_list = list()
        for word in test_word_split:
            result = self.model.get_nearest_neighbors(word)
            similar_letter_dict = defaultdict(list)
            for _,temp in result:
                if not text_proc.remove_other_letter(temp):
                    continue
                for w in text_proc.bad_word:
                    if w in text_proc.separate_reversed(temp):
                        similar_letter_dict[text_proc.separate_reversed(word)].append(w)
            for key,val in similar_letter_dict.items():
                result_list.append(f"{key} ==> 연관 비속어 : {val}")
        return result_list
    

class textProcessing():
    def __init__(self):
        self.stop_word = cf.get_stop_word()
        self.bad_word = cf.get_bad_word()
        self.choseong = [char for char in "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"]
        self.jungseong = [chr(char) for char in range(ord('ㅏ'),ord('ㅣ')+1)]
        self.jongseong = [char for char in "-ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]

    # 한글·공백을 제외한 문자 제거 함수
    def remove_other_letter(self,s):
        return re.sub("[^ㄱ-ㅎ가-힣 ]","",s)
    
    # 형태소 분리 및 불용어 제거 함수
    def tokenize_remove_stop_word(self,s):
        okt = Okt()
        word_list = okt.morphs(s)
        result_list = list()
        for w in word_list:
            if w not in self.stop_word:
                result_list.append(w)
        return result_list

    def separate(self,test_word):
        result_list = list()
        for word in test_word:
            for w in word:
                if ord(w) <=12643:
                    result_list.append(w+'--')
                else:
                    ## 588개 마다 초성이 바뀜.
                    ch1 = (ord(w) - ord('가'))//588
                    ## 중성은 총 28가지 종류
                    ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
                    ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
                    result_list.append(self.choseong[ch1] + self.jungseong[ch2] + self.jongseong[ch3])
        return ''.join(result_list)
    
    def separate_reversed(self,korean_word):
        korean_word = [re.sub('-','',korean_word[i:i+3]) for i in range(0,len(korean_word), 3)]
        result_list = list()
        for word in korean_word:
            result = ord('가')
            if len(word) == 3:
                result += (self.choseong.index(word[0]) * 21 * 28) + (self.jungseong.index(word[1]) * 28) + self.jongseong.index((word[2]))
                result_list.append(chr(result))
            elif len(word) == 2:
                result += (self.choseong.index(word[0]) * 21 * 28) + (self.jungseong.index(word[1]) * 28)
                result_list.append(chr(result))
            else:
                result_list.append(word)
        return ''.join(result_list)
    
    def test_sentence(self,s,fasttext):
        # 1.한글·공백 이외 문자 제거 후, 형태소 분리 및 불용어 제거 수행
        test_word = self.tokenize_remove_stop_word(self.remove_other_letter(s))
        # 2.형태소 별 초성·중성·종성으로 분리
        test_word_split = [self.separate(word) for word in test_word]
        # 3. 분리된 형태소를 3차원 ndarray 벡터로 변환
        test_word_vec = list()
        for i in range(24):
            if i < len(test_word_split):
                test_word_vec.append(fasttext.model[test_word_split[i]])
            else:
                test_word_vec.append(np.array([0]*100))
        test_word_vec = np.array(test_word_vec)
        test_word_vec = test_word_vec.reshape(1,test_word_vec.shape[0],test_word_vec.shape[1])
        return test_word_split,test_word_vec