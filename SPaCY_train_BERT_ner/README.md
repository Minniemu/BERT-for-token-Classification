# 用SPaCY的結果作為訓練資料，重作BERT-ner
* BERT_ner訓練資料格式

>![](https://i.imgur.com/9S0oyWh.png)

* What is POS tagging?
**其實就是詞性的標記**
詞性列表：

CC     coordinatingconjunction 並列連詞

CD     cardinaldigit  純數基數

DT     determiner  限定詞（置於名詞前起限定作用，如 the、some、my 等）

EX     existentialthere (like:"there is"... think of it like "thereexists")   存在句；存現句

FW     foreignword  外來語；外來詞；外文原詞

IN     preposition/subordinating conjunction介詞/從屬連詞；主從連詞；從屬連線詞

JJ     adjective    'big'  形容詞

JJR    adjective, comparative 'bigger' （形容詞或副詞的）比較級形式

JJS    adjective, superlative 'biggest'  （形容詞或副詞的）最高階

LS     listmarker  1)

MD     modal (could, will) 形態的，形式的 , 語氣的；情態的

NN     noun, singular 'desk' 名詞單數形式

NNS    nounplural  'desks'  名詞複數形式

NNP    propernoun, singular     'Harrison' 專有名詞

NNPS  proper noun, plural 'Americans'  專有名詞複數形式

PDT    predeterminer      'all the kids'  前位限定詞

POS    possessiveending  parent's   屬有詞結束語

PRP    personalpronoun   I, he, she  人稱代詞

PRP$  possessive pronoun my, his, hers  物主代詞

RB     adverb very, silently, 副詞非常靜靜地

RBR    adverb,comparative better   （形容詞或副詞的）比較級形式

RBS    adverb,superlative best    （形容詞或副詞的）最高階

RP     particle     give up 小品詞(與動詞構成短語動詞的副詞或介詞)

TO     to    go 'to' the store.

UH     interjection errrrrrrrm  感嘆詞；感嘆語

VB     verb, baseform    take   動詞

VBD    verb, pasttense   took   動詞過去時；過去式

VBG    verb,gerund/present participle taking 動詞動名詞/現在分詞

VBN    verb, pastparticiple     taken 動詞過去分詞

VBP    verb,sing. present, non-3d     take 動詞現在

VBZ    verb, 3rdperson sing. present  takes   動詞第三人稱

WDT    wh-determiner      which 限定詞（置於名詞前起限定作用，如 the、some、my 等）

WP     wh-pronoun   who, what 代詞（代替名詞或名詞片語的單詞）

WP$    possessivewh-pronoun     whose  所有格；屬有詞

WRB    wh-abverb    where, when 副詞

可參考以下文章 : https://medium.com/@muddaprince456/categorizing-and-pos-tagging-with-nltk-python-28f2bc9312c3

* Example:
```
text = "Hello welcome to the world of to learn Categorizing and POS Tagging with NLTK and Python"
[('Hello', 'NNP'), 
	('welcome', 'NN'), 
	('to', 'TO'), 
	('the', 'DT'), 
	('world', 'NN'),
	 ('of', 'IN'), 
	 ('to', 'TO'), 
	 ('learn', 'VB'), 
	 ('Categorizing', 'NNP'), 
	 ('and', 'CC'), 
	 ('POS', 'NNP'), 
	 ('Tagging', 'NNP'),
	  ('with', 'IN'), 
	  ('NLTK', 'NNP'), 
	  ('and', 'CC'), 
	  ('Python', 'NNP')]
```
2021.01.13
---
1. 本來想直接將原始dataset的token放到SPaCY做Tagging就好，結果發現用biluo_tags_from_offsets(doc, entities)這個函式，會自己重新做斷詞 :exploding_head: ，順序會跑掉
2. Solution : 手幹 :anguished: 

