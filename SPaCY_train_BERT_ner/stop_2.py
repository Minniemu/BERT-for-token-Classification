import csv
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from spacy.gold import biluo_tags_from_offsets
import spacy
import json
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 15000000000
file = open('dataset_test.csv',  newline='')
rows = csv.DictReader(file)
#存放原始資料
Sentences = []
Tokens = []
POS = []
temp = []
#複製整個list
def Cloning(list1):
    copy = list1[:]
    return copy
#將csv檔一行一行讀出來
for row in rows:
    if len(row["Sentence #"])==0 : # 如果此行的 Sentence #這個欄位沒有東西
        temp.append(row["Word"]) # 將 Word 放入temp list
        
    else: # 如果 Sentence 本來不為空字串，表為新句子
        Sentences.append(Cloning(temp))  #將temp 裡的token(也就是上一句)放進Sentences
        temp.clear() #清空temp
        temp.append(row["Word"]) # 將 Word 放入新的temp list
#將最後一句也放入Sentences
Sentences.append(Cloning(temp))      
print(Sentences)