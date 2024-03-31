# PROJECT : 언어순화 프로젝트

</br></br>

## 프로젝트 기간 📆

| 날짜 | 업무 내용 |
| --- | --- |
| 2023.02.15 ~ 2023.02.16 | 프로젝트 기획, 주제 선정, 자료 조사 |
| 2023.02.16 ~ 2023.02.18 | 데이터 수집 및 정제(한국어 비속어 • 커뮤니티 게시글 및 댓글 데이터 수집 및 정제) |
| 2023.02.18 ~ 2023.02.20 | NLP Modeling(FastTExt, LSTM) |
| 2023.02.20 ~ 2023.02.21 | PyQt GUI 구현 및 발표자료 작성 |

</br></br>

## 구성원

| 구성원 | 깃허브 주소 | 분담 역할 |
| --- | --- | --- |
| 강재전 | https://github.com/KangJJ63 | 한국어 비속어  • 커뮤니티 게시글 및 댓글 데이터 <br>수집 및 데이터 라벨링, <br> FastText 데이터 임베딩, PyQt 구현 |
| 구성준 | https://github.com/KOO-96 | 커뮤니티 게시글 및 댓글 데이터 <br>수집 및 데이터 라벨링, <br> LSTM 모델 구축 |
| 장윤영 | https://github.com/Jyundev | 한국어 비속어  • 커뮤니티 게시글 및 댓글 데이터 <br>수집 및 데이터 라벨링,  <br> FastText 데이터 임베딩 |

</br>

---

## **Stack**

- **Language** : Python
- **Library & Framework** : Colab, Tensorflow, Fasttext, Sklearn, PyQt
- **Environment**
    
    
    | Env | CPU | GPU | RAM | OS |
    | --- | --- | --- | --- | --- |
    | Local | i3-5005U | RTX-3070Ti | 4G | Window10 |
    | Colab | intel Xeon | T4 GPU | 12G | Ubuntu |

---

### Contents Table

- [프로젝트 개요](https://www.notion.so/ReadMe-582eb35ade15428ebb836d24256dcd66?pvs=21)
- [기대 효과](https://www.notion.so/ReadMe-582eb35ade15428ebb836d24256dcd66?pvs=21)
- [프로젝트 설명](https://www.notion.so/ReadMe-582eb35ade15428ebb836d24256dcd66?pvs=21)
- [모델 설명](https://www.notion.so/ReadMe-582eb35ade15428ebb836d24256dcd66?pvs=21)
- [Dataset](https://www.notion.so/ReadMe-582eb35ade15428ebb836d24256dcd66?pvs=21)
- [Reference](https://www.notion.so/ReadMe-582eb35ade15428ebb836d24256dcd66?pvs=21)

---

### 📝 프로젝트 개요

![alt text](img/hate.png)

*이강인(23·파리 생제르맹)에 대해 커뮤니티에 달린 비난 댓글*

최근 아시안컵에서의 결과에 대해 선수들과 감독에게 지나친 비난이 발생한 일이 있었습니다. 댓글 문제에 대응하기 위해 네이버와 카카오는 2004년 댓글 서비스를 시작한 이후 댓글 개수 제한, 댓글 이력 공개, 댓글 어뷰징 방지 시스템 도입, AI 기반 필터링 적용 및 고도화, 그리고 연예·스포츠 뉴스 댓글 폐지 등 다양한 방식으로 노력하고 있지만, 이러한 노력들이 근본적인 해결책이 되지 못하고 있습니다.

한국의 네티즌들은 댓글을 통해서 정보를 얻는 것보다는 주로 재미와 흥미를 추구한다는 조사 결과가 있습니다. (한국리서치, 2021) 이에 따라 본 프로젝트는 비속어를 탐지하여 순화된 언어로 변환하여 악플을 예방하고, 악플을 다는 행위 자체에 흥미를 잃게 만드는 것을 목적으로 합니다. 이를 통해 건전한 토론과 의견 교환이 가능한 환경을 조성하고자 합니다.

---

### 🛎️ 기대 효과

- 악플 방지
    
    > 비속어를 감지하고 순화된 언어로 변환함으로써 악성 댓글이나 욕설 등의 부적절한 코멘트를 방지합니다.
    > 
- 청정한 댓글 환경 조성
    
    > 악플을 다는 행위에 흥미를 잃게 만듦으로써, 악플러들이 부적절한 언어를 사용하는 것을 억제할 수 있습니다. 이를 통해, 온라인 공간의 사용자들이 더욱 건전하고 즐거운 경험을 할 수 있도록 도와줍니다
    > 
    
    ---
    

### 프로젝트 설명

![alt text](img/flow.png)


**[데이터 전처리]**

- 데이터 별 욕설 여부 라벨링
- 한글·공 외 영어 및 특수문자 제거

<div style="display: flex; justify-content: center;">
  <img src="img/data1.png" alt="Alt text" style="width: 80%; margin: 5px;">
</div>

- 불용어 제거 및 형태소 분리


<div style="display: flex; justify-content: center;">
  <img src="img/data2.png" alt="Alt text" style="width: 80%; margin: 5px;">
</div>

- 형태소 별 초성·중성·종성 분리
    - 댓글 일부분은 “ㅅㅂ”, “ㅄ” 등 초성으로만 이루어진 비속어가 존재합니다. 따라서, 단어가 아닌 자모단위로 분석하기 위해 한국어를 초성·중성·종성으로 분리하였습니다.
    
<div style="display: flex; justify-content: center;">
  <img src="img/data3.png" alt="Alt text" style="width: 80%; margin: 5px;">
</div>

**[모델 개발]**

- 한글의 형태적인 정보를 학습하여, n-gram(n=5) 단어 단위로 임베딩 할 수 있는 FastText모델 생성

<div style="display: flex; justify-content: center;">
  <img src="img/model1.png" alt="Alt text" style="width: 80%; margin: 5px;">
</div>

- 벡터화된 수치 데이터를 사용하여 비속어 여부를 학습하고 예측하기 위한 LSTM 모델을 수립

<div style="display: flex; justify-content: center;">
  <img src="img/model2.png" alt="Alt text" style="width: 80%; margin: 5px;">
</div>

**[모델 비교]**

1. **LSTM v1** 

| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| lstm_52 (LSTM) | (None, 24, 64) | 42240 |
| lstm_53 (LSTM) | (None, 32) | 12416 |
| dense_33 (Dense) | (None, 16) | 528 |
| dropout_19 (Dropout) | (None, 16) | 0 |
| dense_34 (Dense) | (None, 1) | 17 |
- 총 5개의 레이어로 구성됨
- 드롭아웃을 적용하여 과적합 방지

---

1. **LSTM v2** 
- 총 6개의 레이어로 구성됨
    - LSTM v1 모델과 비교하여 LSTM 레이어 추가
- 드롭아웃을 적용하여 과적합 방지

---

1. **GRU (기존 LSTM 모델을 간소화한 버전)**

| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| gru_15 (GRU) | (None, 24, 64) | 31,872 |
| gru_16 (GRU) | (None, 24, 32) | 9,408 |
| gru_17 (GRU) | (None, 16) | 2,400 |
| dense_37 (Dense) | (None, 16) | 272 |
| dropout_21 (Dropout) | (None, 16) | 0 |
| dense_38 (Dense) | (None, 1) | 17 |
- 총 6개의 레이어로 구성됨
- 드롭아웃을 적용하여 과적합 방지

---

**[성능 비교]  Accuracy / Loss** 

![alt text](img/graph.png)

---

### 모델 설명

1. **FastText**

FastText는 Facebook에서 개발한 기술로, 단어를 n-gram의 하위 단어 집합으로 학습하고, 이들을 결합하여 단어의 전체 임베딩을 생성합니다.

![alt text](img/fasetext.png)

FastText를 사용한 이유는 다음과 같습니다.

- 한국어는 조사 등의 불규칙한 형태소로 구성되어 있습니다. 특히 욕설 데이터는 띄어쓰기가 잘 지켜지지 않으며, 때에 따라 초성으로만 이루어진 비속어들이 존재합니다.
- 비슷한 발음 등 어휘 정보를 가진 신조어의 특성에 따라, Word2Vec의 경우 단어가 모델에 없으면 이를 처리하기가 어렵습니다. 반면, FastText 모델은 하위 단어들을 이용하여, 모델에 없는 단어도 유사한 단어들의 정보를 활용하여 임베딩을 생성해 다양한 경우의 수를 고려함으로써 OOV(Out Of Vocabulary)문제를 효과적으로 해결할 수 있습니다.

1. **LSTM**

성준님 베이스만 대충 써주실수 있을까용?!

### Dataset

| Title | link |
| --- | --- |
| 한국어 혐오 데이터셋 | https://github.com/kocohub/korean-hate-speech |
| 일베·오늘의 유머 사이트의 욕설 데이터셋 | https://github.com/2runo/Curse-detection-data |
| 디시인사이드·네이트판 등에서 수집한 욕설 데이터 | - |
| 나무위키 한국어 욕설 정보 | https://namu.wiki/w/%EC%9A%95%EC%84%A4/%ED%95%9C%EA%B5%AD%EC%96%B4 |
| 직접 제작한 불용어 사전 | - |

### Reference