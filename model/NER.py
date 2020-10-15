entities = ['LOC', 'ORG', 'PER', 'NORP', 'DATE', 'O', 'MISC']
entities_rev = {}
for i in range(len(entities)):
    entities_rev[entities[i]] = i
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import spacy
import random

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = None

nlp = spacy.load("en_core_web_sm")


def bert_ner(sequence, tokenizer):
    global model
    if model == None:
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-cased-finetuned-conll03-english")
        model.resize_token_embeddings(len(tokenizer))
    label_list = [
        "O",  # Outside of a named entity
        "MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "MISC",  # Miscellaneous entity
        "PER",  # Beginning of a person's name right after another person's name
        "PER",  # Person's name
        "ORG",  # Beginning of an organisation right after another organisation
        "ORG",  # Organisation
        "LOC",  # Beginning of a location right after another location
        "LOC"  # Location
    ]

    # sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
    #            "close to the Manhattan Bridge."

    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer.encode(sequence, return_tensors="pt")

    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim=2)
    # print(tokens)
    return [label_list[prediction] for token, prediction in zip(tokens, predictions[0].tolist())][1:-1]


def spacy_ner(sequence):
    # sequence = "Apple is looking at buying U.K. startup for $1 billion"

    splited = sequence.split(' ')

    doc = nlp(sequence)

    labels = ['O' for i in range(len(splited))]
    # print(doc.ents)
    # print(splited)
    starts = []
    for i, tok in enumerate(splited):
        if i == 0:
            starts.append(0)
        else:
            starts.append(starts[i - 1] + len(splited[i - 1]) + 1)
    for ent in doc.ents:

        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        label = ent.label_
        if label == "GPE":
            label = "LOC"
        if label == "FAC":
            label = "LOC"
        if label == "EVENT" or label == 'TIME':
            label = "DATE"
        if label in entities:
            sp = ent.text.split(' ')
            # print(sp)
            for ss in sp:
                if ss in splited:
                    idx = splited.index(ss)
                else:
                    idx = len(starts) - 1
                    for i in range(len(starts)):
                        if ent.start_char < starts[i]:
                            idx = i - 1
                            # print(idx, starts[i - 1])
                            break

                labels[idx] = label

    return labels, splited


def spacy_preserve(labels, sp, tokenizer):
    new_labels = []

    # sp = sequence.split(" ")
    for i in range(len(sp)):
        tokenized_word = tokenizer.tokenize(sp[i])
        # print(tokenized_word)
        n_subwords = len(tokenized_word)
        new_labels.extend([labels[i]] * n_subwords)
    return new_labels


def just_spacy_ner(sequence, max_len, tokenizer, by_idex=True):
    spacy_res, sp = spacy_ner(sequence)
    spacy_res = spacy_preserve(spacy_res, sp, tokenizer)
    final = []
    if by_idex:
        for i in range(len(spacy_res)):
            final.append(entities_rev[spacy_res[i]])
        final.extend([entities_rev['O']] * (max_len - len(final)))
    return final


def merged_ner(sequence, max_len, tokenizer, by_idex=True):
    # return dummy_ner(sequence,max_len,tokenizer)

    spacy_res, sp = spacy_ner(sequence)
    spacy_res = spacy_preserve(spacy_res, sp, tokenizer)
    bert_res = bert_ner(sequence, tokenizer)
    final = []
    if by_idex:
        for i in range(len(bert_res)):
            if bert_res[i] == 'O':
                final.append(entities_rev[spacy_res[i]])
            else:
                final.append(entities_rev[bert_res[i]])
        final.extend([entities_rev['O']] * (max_len - len(final)))
    else:
        for i in range(len(bert_res)):
            if bert_res[i] == 'O':
                final.append(spacy_res[i])
            else:
                final.append(bert_res[i])
        final.extend(['O'] * (max_len - len(final)))
    return final


# string = "Apple is looking at buying U.K. startup for $1 billion"
# lb, sp = spacy_ner(string)
# r = spacy_preserve(lb, sp)
# print(r)


def dummy_ner(sequence, max_len, tokenizer):
    # print(len(sequence))

    return [random.randint(0, len(entities) - 1) for i in range(max_len)]
