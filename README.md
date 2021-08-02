# Python Keyphrase Extraction with Korean Language support

## Background
[`pke`](https://github.com/boudinfl/pke) module is the one-stop shop for keyword extraction in Python (except `RAKE`). However, due to its lack of support for languages not included in [`Spacy`](https://spacy.io) language models, it cannot be applied on Korean text. The only change I made here is to add that functionality.

## Requirements
The same requirements of `pke` are applied here as I only tweaked for the preprocessing of text. Additionally, to handle Korean text POS tagging, `konlpy` (for **Unix**-based systems) or `eunjeon` (for **Windows**-based systems) is needed. As I employed `Mecab` module, Mecab-ko should be installed before use.

## Usages
The same syntax applies as the original package.
Refer to the following sample script. (`sample.py`)

``` python
import unsupervised

text = {'ko':'5일 블룸버그통신에 따르면 블랙록, 스테이트 스트리트, JP모간 자산운용, UBS 자산운용 등 세계 최대 규모 자산운용사들은 올해 하반기에도 주식시장이 계속해서 오를 것이란 전망을 제시하고 있다. \
MSCI 전 국가 세계 지수가 올해 12% 상승하며 역대 고점까지 올랐지만 추가 상승 가능성이 높다는 것이다.',
'en':'A mathematical model of ion exchange is considered, allowing for ion exchanger compression in the process of ion exchange. \
Two inverse problems are investigated for this model, unique solvability is proved, and numerical solution methods are proposed. \
The efficiency of the proposed methods is demonstrated by a numerical experiment.'}

extractor = {'TextRank':{'ko':unsupervised.TextRank(),'en':unsupervised.TextRank()},
'TopicRank':{'ko':unsupervised.TopicRank(),'en':unsupervised.TopicRank()},
'YAKE':{'ko':unsupervised.YAKE(),'en':unsupervised.YAKE()}}

for ext in extractor:
    for lang in text:
        extractor[ext][lang].load_document(text[lang],language=lang)
        extractor[ext][lang].candidate_selection()
        extractor[ext][lang].candidate_weighting()
        print(f'{ext} - {lang}')
        print(extractor[ext][lang].get_n_best())
```
