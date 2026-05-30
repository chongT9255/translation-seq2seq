#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import config
from model import TranslationModel
import jieba
from tokenizer import ChineseTokenizer,EnglishTokenizer


def predict_batch(model,inputs,en_tokenizer,device):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入 shape:[batch_size,seq_len]
    :return:  预测结果，shape:[[*,*,*,*],[*,*],[*,*,*]]
    """

    model.eval()
    with torch.no_grad():
        # 编码
        context_vector = model.encoder(inputs)
        # context_vector shape [batch_size,hidden_size]

        # 解码
        batch_size = inputs.shape[0]

        # 准备隐藏状态
        decoder_hidden = context_vector.unsqueeze(0) # decoder_hidden.shape [1,batch_size,hidden_size]

        # decoder_hidden shape [1,batch_size,hidden_size]
        decoder_iutput = torch.full(size=(batch_size,1),fill_value=en_tokenizer.sos_token_index,device= device)

        # 预测结果缓存
        generated = []
        # 记录每个样本是否已经生成结束符
        is_finished = torch.full(size=(batch_size,),fill_value=False,device=device)
        # 自回归生成
        for i in range(config.MAX_SEQ_LEN):
            # 解码
            decoder_output,decoder_hidden = model.decoder(decoder_iutput,decoder_hidden)
            # decoder_output shape [batch_size,1,vocab_size]
            # 保存预测结果
            next_token_indexes = decoder_output.argmax(dim=-1)
            # 更新隐藏状态
            generated.append(next_token_indexes)
            # 更新输入
            decoder_iutput = next_token_indexes
            # 判断是否应该结束
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():
                break
        # 处理预测结果
        # generated shape [tensor([batch_size,1])]
        generated_tensor = torch.cat(generated,dim=-1)
        # generated_tensor.shape [batch_size,seq_len]
        generated_list =  generated_tensor.tolist()
        # 去掉eos之后的token_id
        for index,sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        return generated_list


def predict(text,model,zh_tokenizer,en_tokenizer,device):

    # 1.处理输入
    indexes = zh_tokenizer.encode(text)
    # indexes.shape [seq_len]
    
    input_tensor = torch.tensor([indexes],dtype=torch.long) # 输入的维度是 [batch_size, seq_len]
    input_tensor = input_tensor.to(device)
    # 2.预测
    batch_result = predict_batch(model,input_tensor,en_tokenizer, device)

    return en_tokenizer.decode(batch_result[0])


def run_predict():
    # 资源加载
    print("资源加载中...")
    # 1.确定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.加载词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    print("分词器加载成功")
    # 3.模型
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_idx=zh_tokenizer.pad_token_index,
        en_padding_idx=en_tokenizer.pad_token_index
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "model.pth"))
    print("模型加载成功")

    print("欢迎使用中英翻译系统(输入q或者quit退出)")
    while True:
        user_input = input("中文:")
        if user_input == "q" or user_input == "quit":
            print("欢迎下次再来")
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue

        result  = predict(user_input,model,zh_tokenizer,en_tokenizer,device)
        print("翻译结果：",result)



if __name__ == '__main__':
    # top5_tokens = predict("我们团队")
    # print(top5_tokens)
    run_predict()