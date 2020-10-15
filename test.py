import time

from model.NER import bert_ner, spacy_ner, spacy_preserve
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = 'Apple is looking at buying U.K. startup for $1 billion'
t = time.time()
spacy_res, sp = spacy_ner(sequence)
spacy_res = spacy_preserve(spacy_res, sp, tokenizer)
print(time.time() - t)

bert_ner(sequence, tokenizer)

t = time.time()
bert_res = bert_ner(sequence, tokenizer)
print(time.time() - t)
