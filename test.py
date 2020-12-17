import pandas as pd
import numpy as np
from tqdm import tqdm, trange
#Load the data
import joblib
data = pd.read_csv("ner_dataset.csv", encoding="latin1").fillna(method="ffill")
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
print(sentences[0])
labels = [[s[2] for s in sentence] for sentence in getter.sentences]
print(labels[0])
tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")
tag_values.sort()
tag2idx = {t: i for i, t in enumerate(tag_values)}

print(tag_values)
print(tag2idx)
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel
#import transformers
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

print(torch.__version__)

MAX_LEN = 75
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

import joblib
filename = 'finalized_model.sav'
model = joblib.load(filename)
model.to(device)
model.eval()
test_sentence = """
That day, Yakufu, a 43-year-old ethnic Uyghur, had been freed from a Chinese detention camp and allowed to return home to her three teenage children and aunt and uncle in Xinjiang, western China. It was the first time she'd seen her family in more than 16 months.
"""
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()
with torch.no_grad():
    output = model(input_ids)

label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
print("label_indices",label_indices)
# join bpe split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
print(tokenized_sentence)
print(input_ids)
print(label_indices)
print(tokens)
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)
for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))
