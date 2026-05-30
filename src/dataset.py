#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config

# 1.定义Dataset
class TranslationDataset(Dataset):
    def __init__(self,path):
        # super(InputMethodDataset, self).__init__()
        self.data = pd.read_json(path,orient="records",lines=True).to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]["zh"],dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]["en"],dtype=torch.long)
        return  input_tensor,target_tensor





def collate_fn(batch):
    # print(batch) 一批原始数据
    # batch : 二元组列表[(input_tensor,target_tensor)]
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    # 处理每批数据的长度 + 堆叠
    input_tensor = pad_sequence(input_tensors,batch_first=True,
                                 padding_value=0)
    target_tensor = pad_sequence(target_tensors,batch_first=True,
                                 padding_value=0)
    return input_tensor,target_tensor




# 2.提供一个获取dataloader的方法
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset = TranslationDataset(path)
    # collate_fn 自定义数据堆叠
    return  DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=collate_fn)


if __name__ == '__main__':
    train_loader = get_dataloader()
    test_loader = get_dataloader(train=False)
    print("训练集大小",len(train_loader))
    print("测试集大小",len(test_loader))

    for  inputs,targets in train_loader:
        print("输入",inputs.shape) # (batch_size,seq_len)
        print("目标",targets.shape) # (batch_size,)
        break