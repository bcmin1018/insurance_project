import pandas as pd
import re
from hanspell import spell_checker
import multiprocessing


def clean_text(text):
    try:
        # 순번을 나타내는 특수문자를 숫자로 변경
        text = re.sub(r'①', '1.', text)
        text = re.sub(r'②', '2.', text)
        text = re.sub(r'③', '3.', text)
        text = re.sub(r'④', '4.', text)
        text = re.sub(r'⑤', '5.', text)
        text = re.sub(r'⑥', '6.', text)
        text = re.sub(r'⑦', '7.', text)
        text = re.sub(r'⑧', '8.', text)
        text = re.sub(r'⑨', '9.', text)
        text = re.sub(r'⑩', '10.', text)

        # 한글, 영어, 숫자와 마침표 쉼표 를 제외한 모든 특수문자 제거
        text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-np-wyzA-NP-WYZ0-9.,:()∼ \\s]', '', text)

        # 추가 전처리는 밑에 추가

        # 맞춤법 체크 (500자 단위로 진행), 라이브러리 1회 최대 교정 자리수가 500자 이므로 청크로 나눠서 진행.
        if len(text) > 500:
            chunks = [spell_checker.check(text[i:i + 500]).checked for i in range(0, len(text), 500) ]
            text = ''.join(chunks)
        else:
            text = spell_checker.check(text).checked

    except TypeError as e:
        print(f'에러 발생 텍스트 : {text}')
    return text

if __name__ == '__main__':
    df = pd.read_excel('./data/pdf_data.xlsx')
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    facts_pre = pool.map(clean_text, df['facts'])
    args_pre = pool.map(clean_text, df['args'])
    df['facts_pre'] = facts_pre
    df['args_pre'] = args_pre
    df.to_excel('pdf_data(전처리).xlsx', index=False)
