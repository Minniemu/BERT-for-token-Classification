#Setup the Bert model for finetuning
import transformers
from transformers import BertForTokenClassification, AdamW

print(transformers.__version__)
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)
model.cuda();
