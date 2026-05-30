from functorch import dim
from sklearn.externals.array_api_compat import torch
from torch import nn
import config


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx # 填充的索引,减少padding_token的影响
                                      )
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True
                          ) # 单层单向

    def forward(self,x):
        # x.shape [batch_size,seq_len]
        embed = self.embedding(x)
        # embed.shape [batch_size,seq_len,embedding_dim]
        output,_ = self.gru(embed)
        # output.shape [batch_size,seq_len,hidden_size]
        # 取最后一个隐藏状态
        lengths = (x!=self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = output[torch.arange(output.shape[0]),lengths - 1]
        # last_hidden_state.shape [batch_size,hidden_size]
        return last_hidden_state

class TranslationDecoder(nn.Module):

    def __init__(self,vocab_size,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx
                                      )
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True
                          )
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                            out_features=vocab_size # 英文的词表
                            )
    # 前向传播，每次生成一个token，后续进行循环调用
    def forward(self,x,hidden_0):
        # x.shape [batch_size,1]  ==》 seq_len =1
        # hidden_0.shape [1,batch_size,hidden_size]
        embed = self.embedding(x)
        # embed.shape [batch_size,1,embedding_dim]
        output,hidden_n = self.gru(embed,hidden_0)
        # output.shape [batch_size,1,hidden_size]
        logits = self.linear(output)
        # logits.shape [batch_size,1,vocab_size]
        return logits,hidden_n


class TranslationModel(nn.Module):
    def __init__(self,zh_vocab_size,en_vocab_size,zh_padding_idx,en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size,padding_idx=zh_padding_idx)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size,padding_idx=en_padding_idx)

    