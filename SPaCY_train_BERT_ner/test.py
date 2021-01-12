import spacy
from spacy.gold import biluo_tags_from_offsets
nlp = spacy.load("en_core_web_sm")

text = "Trump has not yet received the vaccine and won't be administered one until it is recommended by the White House medical team, a White House official previously told CNN."
doc = nlp(text)
print("Entities before adding new entity:",doc.ents)

entities = []
for ent in doc.ents:
    entities.append((ent.start_char, ent.end_char, ent.label_))
    print(ent,ent.start_char, ent.end_char, ent.label_)
print("BILUO before adding new entity:", biluo_tags_from_offsets(doc, entities))

