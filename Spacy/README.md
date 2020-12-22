---
title: 'Project documentation template'
disqus: hackmd
---

Spacy 教學
===


程式碼
---
1. 因為過程中會用到 en_core_web_sm，記得先下載
```
python3 -m spacy download en_core_web_sm
```
2. 程式碼如下:
```
import spacy
import json
nlp = spacy.load("en_core_web_sm")
#放你想放的文章
text = '''That day, Yakufu, a 43-year-old ethnic Uyghur, had been freed from a Chinese detention camp and allowed to return home to her three teenage children and aunt and uncle in Xinjiang, western China. It was the first time she'd seen her family in more than 16 months.'''

doc = nlp(text)

tag = ['GPE','ORG','PERSON','PERCENT','NORP']
res = {"article": text , "answers":{"ans_detail":[]}}
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    if ent.label_ in tag:
        tmp = {"tag" : ent.text, "start_at" : ent.start_char, "end_at" : ent.end_char}
        res["answers"]["ans_detail"].append(tmp)

print(json.dumps(res))
```
3. Result : 
```
{"article": "That day, Yakufu, a 43-year-old ethnic Uyghur, had been freed from a Chinese detention camp and allowed to return home to her three teenage children and aunt and uncle in Xinjiang, western China. It was the first time she'd seen her family in more than 16 months.\n\n", 
"answers": 
{"ans_detail": [
{"tag": "Yakufu", "start_at": 10, "end_at": 16}, 
{"tag": "Uyghur", "start_at": 39, "end_at": 45}, 
{"tag": "Chinese", "start_at": 69, "end_at": 76}, 
{"tag": "Xinjiang", "start_at": 171, "end_at": 179}, 
{"tag": "China", "start_at": 189, "end_at": 194}]}}

```




