import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import requests

from collections import Counter
from tabulate import tabulate

import torch
from torch.utils.data import TensorDataset, DataLoader

# 결측치&중복치가 존재하면 제거하는 함수
def df_cleaning(dataframe, column):
    # 결측치가 존재하면 결측치 제거 수행
    clean_df = dataframe.dropna(how='any')
    # 특정 컬럼을 기준으로 중복치 제거
    clean_df = clean_df.drop_duplicates(
        subset=column, keep='first')
    
    return clean_df

# document 컬럼에 대한 워드 토크나이징 수행
def tokenize(x_data, word_tokenizer):
    tokenized_sentences = list()

    for sent in tqdm(x_data):
        tokenized_sent = word_tokenizer.morphs(sent)
        tokenized_sentences.append(tokenized_sent)

    return tokenized_sentences


# 웹상에 업로드되어 있는 stopword_list 다운로드
def download_stopword_list(url):
    response = requests.get(url)
    stopword_list = []
    # 요청이 성공했는지 확인 후 파일 저장 및 불용어 리스트 읽기
    if response.status_code == 200:
        # 파일 저장 (쓰기 모드)
        with open('kr_stopword_list.txt', 'w', encoding='utf-8') as file:
            file.write(response.text)

        # 파일 읽기 (읽기 모드)
        with open('kr_stopword_list.txt', 'r', encoding='utf-8') as file:
            stopword_list = file.read().splitlines()
    else:
        print("파일 다운로드에 실패했습니다. 상태 코드:", response.status_code)

    return stopword_list

# 리스트 컴프리핸션을 적용하여 빠르게 불용어를 제거하는 함수
def remove_stopword(tokenized_data, stopword):
    return [[word for word in sent if word not in stopword] 
            for sent in tokenized_data]

# 토큰화된 데이터와 클리닝(불용어 제거)된 토큰 데이터 비교
def val_token(token, cl_token, sample_idx):
    print(f'불용어 제거 전: {token[sample_idx]}')
    print(f'불용어 제거 후: {cl_token[sample_idx]}')


ko_unicode = [0xAC00, 0xD7A3] #한글 유니코드 시작 '가', 끝 '힣'

cho_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
jung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
jong_list = ['_'] + ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

cjj_list = [cho_list, jung_list, jong_list]


def word_to_jamo(token):
    # 유니코드상 한글이 아닌 토큰데이터가 있으면 자모분리 안함
    for word in token:
        if ord(word) < ko_unicode[0] or ord(word) > ko_unicode[1]:
            return token
    
    jamo_str = ''
    for koword in token:
        # 한글단어(koword)를 초성/중성/종성으로 분리
        # 한글 유니코드 : (초성*(21*28) + 중성*28) + 종성 + 0xAC00
        ko_code = ord(koword) - ko_unicode[0]
        
        cho_idx = ko_code // (21 * 28)
        jung_idx = (ko_code % (21 * 28)) // 28
        jong_idx = ko_code % 28
        
        cho = cjj_list[0][cho_idx]
        jung = cjj_list[1][jung_idx]
        jong = cjj_list[2][jong_idx]
        
        jamo_str += cho + jung + jong

    return jamo_str


def jamo_to_word(jamo_seq):
    word_str = ""
    # 자모로 분리된 데이터는 무조건 3의 배수가 됨
    # 즉, 길이가 3의 배수가 아닌건 원복 처리 안함
    if len(jamo_seq) % 3 != 0:
        return jamo_seq
    else:
        # 자모 시퀀스를 3개 단위로 끊어서 리스트화
        jamo_list = [jamo_seq[i:i+3] for i in range(0, len(jamo_seq), 3)]

        # 자모 시퀀스는 [초성, 중성, 종성]의 jamo_word로 리스트화됨
        for jamo_word in jamo_list:
            # 단어를 구성하는 jamo가 
            # 초/중/종성 리스트에 없는 이상한 단어면 걸러냄
            for idx, jamo in enumerate(jamo_word):
                if jamo not in cjj_list[idx]:
                    return jamo_seq

            # 자모의 초/중/종성의 index값을 구함
            cho_idx = cjj_list[0].index(jamo_word[0])
            jung_idx = cjj_list[1].index(jamo_word[1])
            jong_idx = cjj_list[2].index(jamo_word[2])
            
            # 한글 유니코드 : (초성*(21*28) + 중성*28) + 종성 + 0xAC00
            ko_code = cho_idx*(21*28) + jung_idx*28 + jong_idx + ko_unicode[0]
            # 유니코드 기반으로 한글 단어 복원
            ko_word = chr(ko_code)
            
            word_str += ko_word
    
    return word_str

# 토큰화처리 + 데이터 클리닝을 마친 한글 데이터를 자모 단위로 분해
def decompose_jamo(cl_x):
    jamo_x_data = []

    for sent in tqdm(cl_x):
        temp = []
        for word in sent:
            jamo = word_to_jamo(word)
            temp.append(jamo)
        jamo_x_data.append(temp)
    
    return jamo_x_data

# 훈련, 검증, 평가로 분리한 데이터셋이 올바르게 클래스 비율대로 나뉘어졌는지 확인하는 함수
def val_class_ratio(x, y, content):
    # 클래스 비율 계산하는 함수
    def class_ratio(label):
        lable_counts = Counter(label)
        ratio = list(lable_counts.values())[0] / sum(list(lable_counts.values()))
        return f"{ratio*100:.2f}%"
    print(f"{content} 데이터셋 개수: {len(x)}, 클래스 비율: {class_ratio(y)}")


# 초기 단어장이 생성되었으면, 해당 단어장의 샘플을 확인하는 코드
def val_raw_vocab(raw_vocab, sample):
    print(f"총 단어 종류: {len(raw_vocab)}")

    vocab_sample = list(raw_vocab.items())[:sample]
    
    # 단어와 등장 횟수를 담은 리스트 생성
    table_data = [[word, freq] for word, freq in vocab_sample]
    
    # tabulate를 이용하여 테이블 출력
    print(tabulate(table_data, headers=["단어", "등장횟수"], 
                   tablefmt="grid", stralign="left"))
    

def set_rare_vocab(raw_vocab, th, report=False):
    tot_vocab_cnt = len(raw_vocab) # 초기 단어장의 총 단어개수
    rare_vocab_cnt = 0 # 초기 단어장 중 희소단어 개수를 저장할 변수
    # 단어 당 등장비율, 희소단어의 등장비율을 계산하기 위한 변수
    tot_freq, rare_freq = 0, 0

    for key, val in raw_vocab.items():
        # 전체 단어의 등장빈도를 계산하는 코드
        tot_freq += val

        # 희소단어의 등장빈도를 계산하는 코드
        if (val < th):
            rare_vocab_cnt += 1
            rare_freq += val

    rare_prop = rare_vocab_cnt / tot_vocab_cnt
    rare_freq_prop = rare_freq / tot_freq

    if report:
        print(f'초기 단어장에 기록된 단어 종류 : {tot_vocab_cnt}')
        print(f'초기 단어장 중 희소단어({th-1}번 미만 등장) 종류 : {rare_vocab_cnt}')
        print(f'초기 단어장 내 희소단어 비율 : {rare_prop*100:.2f} %')
        print(f'전체 단어 중 희소단어 등장 비율 : {rare_freq_prop*100:.2f} %')

    # 총 단어 개수 및 희소단어 개수는 반환한다
    return tot_vocab_cnt, rare_vocab_cnt


def val_vocab_info(word_to_idx, vocab):
    print(f"단어집합(vocab)은 word_to_idx를 통해서")
    print(f"[단어 : idx]의 {type(word_to_idx)}타입이 되고")
    print(f"스페셜 토큰 '<PAD>', '<UNK>'을 포함하여")
    print(f"총 관리되는 단어 '{len(vocab)}' -> '{len(word_to_idx)}'가 됨")



# 단어를 정수 인덱싱 규칙으로 정수 인덱싱 수행하기
def text_to_sequences(tokenized_data, word_to_idx):
    encoded_data = [] #리턴해야할 정수 인코딩 결과값

    if '<UNK>' not in word_to_idx:
        print(f'word_to_idx에 스페셜 토큰 있는지 확인')
        return tokenized_data

    for sent in tokenized_data:
        idx_sequence = [] #단어장 리스트에서 idx를 찾아서 여기에 입력
        
        for word in sent:
            try: #word_to_idx에서 단어를 찾은 뒤 해당 단어의 인덱스(숫자)를 입력
                idx_sequence.append(word_to_idx[word])
            except KeyError: # word_to_idx 딕셔너리에 없는 키(단어)등장시 UNK로 인덱싱
                idx_sequence.append(word_to_idx['<UNK>'])
        
        #문장 내 단어를 모두 정수로 변환한 후에 이를 리턴값(리스트)에 입력
        encoded_data.append(idx_sequence)
    
    return encoded_data



def val_encode_decode(sample_idx, idx_to_word, token, encode, content):
    # 임의의 샘플(sample_idx)를 바탕으로
    # 토큰화된 데이터(token : 원본)
    # 해당 토큰을 정수인코딩 한 결과(encode)
    # 이 정수인코딩을 다시 복원(decode)한 결과의 비교
    decode_sample = [idx_to_word[word] for word in encode[sample_idx]]

    tok_restore_sample = []
    decode_restore_sample = []
    if content == 'Kor':
        for jamo in token[sample_idx]:
            word = jamo_to_word(jamo)
            tok_restore_sample.append(word)
        for jamo in decode_sample:
            word = jamo_to_word(jamo)
            decode_restore_sample.append(word)

    if content == 'Kor':
        print(f'토큰화 데이터의 원문: {tok_restore_sample}')
    print(f'토크나이징만 된 문장: {token[sample_idx]}')
    print(f'정수인코딩 된 결과값: {encode[sample_idx]}')
    print(f'디코드로 복원한 문장: {decode_sample}')
    if content == 'Kor':
        print(f'디코드 문장 자모복원: {decode_restore_sample}')

    # 정수 인코딩 확인
    correct_encoding = True  # 인코딩이 올바른지 확인하기 위한 플래그

    # 디코드된 결과와 토큰화된 결과 비교
    for token_word, decoded_word in zip(token[sample_idx], decode_sample):
        # 디코드된 결과가 <UNK>일 경우, 해당 단어는 비교하지 않음
        if decoded_word != "<UNK>":
            if token_word != decoded_word:
                correct_encoding = False
                break

    if correct_encoding:
        print(f"정수인코딩은 올바르게 진행됨")
    else:
        print(f'정수인코딩에 문제가 있음')



def set_sent_pad(encode_x, context_length, report=False):
    if report:
        print(f'훈련 데이터셋 최대 길이: {max(len(exam) for exam in encode_x)}')
        print(f'훈련 데이터셋 평균 길이: {sum(map(len, encode_x)) / len(encode_x):.2f}')

        plt.hist([len(exam) for exam in encode_x], bins=50)
        plt.xlabel('length of samples')
        plt.ylabel('number of samples')
        plt.show()

    def below_th_len(encode_x, context_length):
        cnt = 0
        for sent in encode_x:
            if(len(sent) <= context_length):
                cnt +=1
        
        print(f'데이터셋 문장 길이가 {context_length} 이하 데이터 비율: '+
              f'{cnt / len(encode_x)*100:.2f}%')
        
    below_th_len(encode_x, context_length)



# x_data 항목을 문장패딩하기 위한 코드
def pad_seq_x(x_data, max_len):
    features = np.zeros((len(x_data), max_len), dtype=int)

    for idx, sent in enumerate(x_data):
        if len(sent) != 0: #예외처리구문
            features[idx, :len(sent)] = np.array(sent)[:max_len]
    
    return features

# 마지막으로 문장패팅이 잘 수행되었는지 검증하는 함수
def val_pad_shape(x, content):
    print(f"{content}용 정수(원핫)인코딩 shape: {x.shape}")


# 데이터 전처리가 완료된 항목을 텐서 데이터 로더로 변환하는 함수
def set_dataloader(x_data, y_label, bs, content=None, report=False):
    # 입력된 데이터를 텐서 자료형 -> 데이터셋 -> 데이터로더로 변환
    tensor_x = torch.tensor(x_data, dtype=torch.int64)
    tensor_y = torch.tensor(y_label, dtype=torch.int64)

    dataset = TensorDataset(tensor_x, tensor_y)

    shuffle = False
    if content == '훈련': # 훈련 시에만 셔플(섞기)를 수행
        shuffle = True
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bs)

    if report: #결과물을 확인하고 싶을 때 아래 코드 동작
        print(f"{content}용 X(인코딩)데이터 크기: {list(tensor_x.shape)}")
        print(f"{content}용 Y(Label)데이터 크기: {list(tensor_y.shape)}")

    return dataloader