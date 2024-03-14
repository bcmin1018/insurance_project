import requests
import pandas as pd
import time
import os
import re
from bs4 import BeautifulSoup

page_url = 'https://wpwsyn.tistory.com/category/%3C%3C%EB%B3%B4%ED%97%98%EA%B4%80%EB%A0%A8%3E%3E/%EB%B3%B4%ED%97%98%EB%B6%84%EC%9F%81%EC%A1%B0%EC%A0%95%EC%82%AC%EB%A1%80?page={}'
for page in range(1, 27):
    # time.sleep(10)
    my_dict = {}
    urls = []
    contents = []
    print('%s page start ' % page)
    res = requests.get(page_url.format(page))
    soup = BeautifulSoup(res.content, 'html.parser')

    # article 태그는 각각의 블로그 단위로 존재
    articles = soup.find_all('article')
    for article in articles:
        # href 태그에 각 블로그 number가 존재
        href = article.find('a')['href']
        url = 'https://wpwsyn.tistory.com' + href
        print('현재 크롤링 URL ', url)
        urls.append(url)
        # time.sleep(3)
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        # p_tags = soup.find_all('p')
        # content = [p_tag.text for p_tag in p_tags]
        div_tags = soup.find_all('div')
        content = [div_tag.text for div_tag in div_tags]
        content = '\n'.join(content)
        contents.append(content)
    print('%s page complete ' % page)
    my_dict['url'] = urls
    my_dict['content'] = contents

    # 데이터 프레임에 저장 (add 방식)
    df = pd.DataFrame(my_dict)
    df.to_csv('my_csv_file.csv', mode='a', header=not os.path.exists('my_csv_file.csv'), index=False)