
# PROJECT : 언어순화 프로젝트 
</br></br>

## 프로젝트 기간 📆

|날짜|업무 내용|
|:--:|:--:|
|2023.02.07 ~ 2023.02.14|프로젝트 기획, 주제 선정, 자료 조사|
|2023.02.15 ~ 2023.02.18|데이터 수집 및 정제(한국어 비속어 • 커뮤니티 게시글 및 댓글 데이터 수집 및 정제)|
|2023.02.18 ~ 2023.02.21|NLP Modeling(FastTExt, LSTM)|
|2023.02.21 ~ 2023.02.23|PyQt GUI 구현 및 발표자료 작성|

</br></br>

## 구성원 

|구성원|깃허브 주소|분담 역할|
|:---:|:--:|:--:|
|강재전|[Git](https://github.com/KangJJ63)|한국어 비속어  • 커뮤니티 게시글 및 댓글 데이터 <br>수집 및 데이터 라벨링, <br> FastText 데이터 임베딩, PyQt 구현|
|구성준|[Git](https://github.com/KOO-96)|커뮤니티 게시글 및 댓글 데이터 <br>수집 및 데이터 라벨링, <br> LSTM 모델 구축|
|장윤영|[Git](https://github.com/Jyundev)|한국어 비속어  • 커뮤니티 게시글 및 댓글 데이터 <br>수집 및 데이터 라벨링,  <br> FastText 데이터 임베딩

</br>  

---

## Enviroment

| Env |CPU | GPU | RAM | OS 
|:--:|:--:|:--:|:--:|:--:|
| Local | i3-5005U | RTX-3070Ti | 4G| Window10 |
| Colab | intel Xeon | T4 GPU | 12G | Ubuntu |


---

### Contents Table
- [프로젝트 개요](#프로젝트-개요)
- [기대 효과](#기대-효과)
- [프로젝트 설명](#프로젝트-설명)  
- [모델 설명](#about-model)
- [Dataset](#Dataset)
- [Reference](#Reference) 

---

### 프로젝트 개요

<div style="display: flex; justify-content: center;">
  <img src="hate_comment2.png" alt="Alt text" style="width: 100%; margin:10px;">
</div>

최근 아시안컵에서의 결과에 대해 선수들과 감독에게 지나친 비난이 발생한 일이 있었습니다. 댓글 문제에 대응하기 위해 네이버와 카카오는 2004년 댓글 서비스를 시작한 이후 댓글 개수 제한, 댓글 이력 공개, 댓글 어뷰징 방지 시스템 도입, AI 기반 필터링 적용 및 고도화, 그리고 연예·스포츠 뉴스 댓글 폐지 등 다양한 방식으로 노력하고 있지만, 이러한 노력들이 근본적인 해결책이 되지 못하고 있습니다.

한국의 네티즌들은 댓글을 통해서 정보를 얻는 것보다는 주로 재미와 흥미를 추구한다는 조사 결과가 있습니다. (한국리서치, 2021) 이에 따라 본 프로젝트는 비속어를 탐지하여 순화된 언어로 변환하여 악플을 예방하고, 악플을 다는 행위 자체에 흥미를 잃게 만드는 것을 목적으로 합니다. 이를 통해 건전한 토론과 의견 교환이 가능한 환경을 조성하고자 합니다.

### 기대 효과

### 프로젝트 설명

### 모델 설명
주요 단어 임베딩 기법으로는, Word2Vec 방식과 FastText 방식이 있습니다.
Word2Vec은 주변 단어들을 고려하여 해당 단어의 임베딩을 학습하는 신경망 기반의 모델입니다.
FastText는 Facebook에서 개발한 기술로, 단어를 n-gram의 하위 단어 집합으로 학습하고, 이들을 결합하여 단어의 전체 임베딩을 생성합니다.



단어 임베딩을 생성하기 위해 해당 프로젝트에서는 FastText방식을 사용하였습니다.

FastText를 사용한 이유는 다음과 같습니다.

- 한국어는 조사 등의 불규칙한 형태소로 구성되어 있습니다. 특히 욕설 데이터는 띄어쓰기가 잘 지켜지지 않으며, 때에 따라 초성으로만 이루어진 비속어들이 존재합니다. 
- 비슷한 발음 등 어휘 정보를 가진 신조어의 특성에 따라, Word2Vec의 경우 단어가 모델에 없으면 이를 처리하기가 어렵습니다. 반면, FastText 모델은 하위 단어들을 이용하여, 모델에 없는 단어도 유사한 단어들의 정보를 활용하여 임베딩을 생성합니다. 이렇듯 다양한 경우의 수를 고려함으로써 OOV(Out Of Vocabulary)문제를 효과적으로 해결할 수 있습니다.



### Dataset

### Reference


|Reference|Git|paper_link|
|:--:|:--:|:--:|
|Swear Word Detection Method Using The Word Embedding and LSTM || [paper](https://oak.chosun.ac.kr/bitstream/2020.oak/16586/2/%EB%8B%A8%EC%96%B4%20%EC%9E%84%EB%B2%A0%EB%94%A9%EA%B3%BC%20LSTM%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%B9%84%EC%86%8D%EC%96%B4%20%ED%8C%90%EB%B3%84%20%EB%B0%A9%EB%B2%95.pdf)|
|FastText:Library for efficient text classification and representation learning|[FastText](https://github.com/facebookresearch/fastText) | [paper](https://fasttext.cc/)|
|The Unreasonable Effectiveness of Recurrent Neural Networks|[char-rnn](https://github.com/karpathy/char-rnn) | [paper](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) |
|