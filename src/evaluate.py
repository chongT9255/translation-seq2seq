#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import config
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer,EnglishTokenizer

def evaluate(model, test_dataloader, device,en_tokenizer):
    model.eval()
    # id不需要转为token
    predictions = []
    # reference_id shape [[*,*,*,*,*],[*,*,*,*],[*,*,*,]]
    references = []
    # reference shape [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*,]]]
    for inputs, targets in tqdm(test_dataloader, desc="评估"):
        inputs = inputs.to(device)
        # inputs.shape == [batch_size, seq_len]
        targets = targets.tolist()
        # targets:[[sos,*,*,*,*,eos],[sos,*,*,*,eos,pad],[sos,*,*,eos,pad,pad]]

        batch_results = predict_batch(model,inputs,en_tokenizer, device)
        # batch_results: [[*,*,*,*,*],[*,*,*,*],[*,*,*,]]

        # 对targets去除特殊符号

        predictions.extend(batch_results) # 把元素一个一个拿出来放入
        references.extend([[target[1:target.index(en_tokenizer.eos_token_index)]] for target in targets])

    bleu = corpus_bleu(references,predictions)
    return bleu




def run_evaluate():
    # 1.确定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.加载词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    # 3.模型
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_idx=zh_tokenizer.pad_token_index,
        en_padding_idx=en_tokenizer.pad_token_index
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "model.pth"))

    # 4.数据集
    test_dataloader = get_dataloader(train=False)

    # 5.评估模型
    bleu = evaluate(model,test_dataloader,device,en_tokenizer)
    print("评估结果：")
    print(f"bleu:{bleu}")


if __name__ == '__main__':
    run_evaluate()