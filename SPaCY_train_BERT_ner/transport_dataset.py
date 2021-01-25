import csv
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from spacy.tokenizer import Tokenizer
from spacy.gold import biluo_tags_from_offsets
import spacy
import json
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 15000000000
file = open('ner_dataset.csv',  newline='', encoding='utf-8', errors='ignore')
rows = csv.DictReader(file)
#存放原始資料
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
        Tokens.append(Cloning(temp))  #將temp 裡的token(也就是上一句)放進Sentences
        temp.clear() #清空temp
        temp.append(row["Word"]) # 將 Word 放入新的temp list
#將最後一句也放入Sentences
Tokens.append(Cloning(temp))      
Tokens.remove([])
print(Tokens)



Sentences = []  #存放整個句子
s = 0
with open( 'output.csv', 'w', newline='') as csvfile:
  # 定義欄位
    fieldnames = ['Sentence #', 'Word', 'POS', 'Tag']

  # 將 dictionary 寫入 CSV 檔
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  # 寫入第一列的欄位名稱
    writer.writeheader()

    #Detokenize
    for token in Tokens:
        s = s+1
        print("Sentence:",s)
        sentence = TreebankWordDetokenizer().detokenize(token) #組成一個句子
        Sentences.append(sentence)
        doc = nlp(sentence)
        entities = []
        tokens = []
        pos = []
        label = []
        for token in doc :
            tokens.append(token.text) #斷詞
            pos.append(token.tag_) #POS 
        for i in range(0,len(tokens)): #先將label 初始化為全 O
            label.append('O')
        ents = [e.text for e in doc.ents] #放關鍵字
        l = [ e.label_ for e in doc.ents] #放LABEL
        for i in range(0,len(ents)): 
            d = nlp(ents[i]) # d 用來存放抓到的關鍵字
            t = [] #t 用來存放斷詞的結果
            for to in d:
                t.append(to.text) #斷詞
                print(t)
                try :
                    index = tokens.index(t[0]) # 搜尋關鍵字在句子中的位置
                except ValueError : 
                    index = 0
                    pass

                t_len = len(t) #計算關鍵字長度
                if t_len == 1: #如果關鍵字只有一個字
                    label[index] = 'B-'+l[i]
                else : #處理關鍵字超過一個字的
                    for j in range(0,t_len): 
                        if j == 0:
                            label[index] = 'B-'+l[i] #只有第一個字是以B-開頭
                        else:
                            label[index+j] = 'I-'+l[i] #其他都是以I-再加上Label

        '''#biluo_tag
        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))
        tags = biluo_tags_from_offsets(doc, entities)'''

        #write in the file
        length = len(tokens)
        for i in range(0,length):
            if i == 0:
                string = "Sentence: "+str(s)
                writer.writerow({'Sentence #':string, 'Word': tokens[i], 'POS':pos[i], 'Tag':label[i]})
            else :
                writer.writerow({'Sentence #':' ', 'Word': tokens[i], 'POS':pos[i], 'Tag':label[i]})
        #write sentence and tag
        '''print("Tokens:",tokens)
        print("Tokens's length=",len(tokens))
        print("POS:",pos)
        print("POS's length=",len(pos))
        print("Tags : ",tags)
        print("Tags' length=",len(tags))
        print()'''

#print(Sentences)
#print(entities)