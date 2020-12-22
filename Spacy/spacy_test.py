import spacy
import json
nlp = spacy.load("en_core_web_sm")
text = '''That day, Yakufu, a 43-year-old ethnic Uyghur, had been freed from a Chinese detention camp and allowed to return home to her three teenage children and aunt and uncle in Xinjiang, western China. It was the first time she'd seen her family in more than 16 months.

'''

doc = nlp(text)

tag = ['GPE','ORG','PERSON','PERCENT','NORP']
res = {"article": text , "answers":{"ans_detail":[]}}
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    if ent.label_ in tag:
        tmp = {"tag" : ent.text, "start_at" : ent.start_char, "end_at" : ent.end_char}
        res["answers"]["ans_detail"].append(tmp)

print(json.dumps(res))