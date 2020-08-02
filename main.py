# %%
import torch
import string

from transformers import \
    AlbertTokenizer, AlbertForMaskedLM,\
    DistilBertTokenizer, DistilBertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
albert_model = AlbertForMaskedLM.from_pretrained('albert-base-v2').eval()

albert_large_tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
albert_large_model = AlbertForMaskedLM.from_pretrained('albert-large-v2').eval()

distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
distilbert_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased').eval()

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-large').eval()


top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= ALBERT =================================
    input_ids, mask_idx = encode(albert_tokenizer, text_sentence.lower())
    with torch.no_grad():
        predict = albert_model(input_ids)[0]
    albert = decode(albert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ALBERT-large =================================
    input_ids, mask_idx = encode(albert_large_tokenizer, text_sentence.lower())
    with torch.no_grad():
        predict = albert_large_model(input_ids)[0]
    albert_large = decode(albert_large_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= DistillBERT =================================
    input_ids, mask_idx = encode(distilbert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = distilbert_model(input_ids)[0]
    distilbert = decode(distilbert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= RoBERTa Large =================================
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)


    return {
        'albert': albert,
        'albert_large': albert_large,
        'distilbert': distilbert,
        'roberta': roberta
    }
