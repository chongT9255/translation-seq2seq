#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer,EnglishTokenizer
import config
from torch import nn
from src.model import TranslationModel
from tqdm import tqdm

def train_one_epoch(model, datalodader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(datalodader):
        # inputs.shape [batch_size,src_seq_len]
        # targets.shape [batch_size,tgt_seq_len]
        encoder_inputs, targets = inputs.to(device), targets.to(device)
        decoder_inputs = targets[:, :-1] # decoder_inputs.shape [batch_size,seq_len]
        decoder_targets = targets[:, 1:] # decoder_targets.shape [batch_size,seq_len]
        # 前向传播
        # 编码阶段
        context_vector = model.encoder(encoder_inputs)
        # context_vector.shape [batch_size,hidden_size]

        # 解码阶段
        decoder_hidden = context_vector.unsqueeze(0)
        # decoder_hidden.shape [1,batch_size,hidden_size]

        decoder_outputs = []
        seq_len = decoder_targets.shape[1]
        for i in range(seq_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1) # decoder_input.shape [batch_size,1]
            decoder_output,decoder_hidden = model.decoder(decoder_input,decoder_hidden)
            # decoder_output.shape [batch_size,1,vocab_size]
            decoder_outputs.append(decoder_output)

        # decoder_outputs:[tensor([batch_size,1,vocab_size])] -> batch_size*seq_len = N -> [N,vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape [batch_size,seq_len,vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_outputs.shape [N,vocab_size]

        # decoder_targets:[batch_size,seq_len] -> batch_size*seq_len = N -> [N]
        decoder_targets = decoder_targets.reshape(-1)

        # 算损失
        loss = loss_fn(decoder_outputs, decoder_targets)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(datalodader)


def train():
    # 1.设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.数据
    datalodader = get_dataloader()
    # 3.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR/'en_vocab.txt')

    # 4.模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_idx=zh_tokenizer.pad_token_index,
                             en_padding_idx=en_tokenizer.pad_token_index).to(device)

    # 5.损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    # 6.优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 7.TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOG_DIR/time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float("inf")
    for epoch in range(1, 1 + config.EPOCHS):
        print(f"========= Epoch:{epoch} =========  ")
        loss = train_one_epoch(model, datalodader, loss_fn, optimizer, device)
        print(f"Loss:{loss:.4f}")

        # 8.TensorBoard
        writer.add_scalar("Loss", loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR/'model.pth')
            print("保存模型")


if __name__ == '__main__':
    train()