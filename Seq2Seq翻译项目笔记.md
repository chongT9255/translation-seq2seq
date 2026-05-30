# Seq2Seq 中英翻译项目详解

> 一份面向新手的、逐文件拆解的 Seq2Seq（序列到序列）机器翻译项目笔记。

---

## 目录

1. [项目概览](#1-项目概览)
2. [项目结构](#2-项目结构)
3. [配置文件 — config.py](#3-配置文件--configpy)
4. [分词器 — tokenizer.py](#4-分词器--tokenizerpy)
5. [数据预处理 — process.py](#5-数据预处理--processpy)
6. [数据集与加载器 — dataset.py](#6-数据集与加载器--datasetpy)
7. [模型定义 — model.py](#7-模型定义--modelpy)
8. [训练脚本 — train.py](#8-训练脚本--trainpy)
9. [推理/预测 — predict.py](#9-推理预测--predictpy)
10. [评估脚本 — evaluate.py](#10-评估脚本--evaluatepy)
11. [核心概念速查](#11-核心概念速查)

---

## 1. 项目概览

**任务**：中文 → 英文翻译（字符级中文切分，词级英文切分）

**模型架构**：经典的 Seq2Seq = Encoder（编码器）+ Decoder（解码器），均使用单层单向 GRU。

**数据流**：

```
原始平行语料 (cmn.txt)
    │
    ▼
process.py ─── 构建词表 + tokenize → train.jsonl / test.jsonl
    │
    ▼
dataset.py ─── 封装为 PyTorch Dataset → DataLoader
    │
    ▼
train.py ─── Encoder → Decoder → CrossEntropyLoss → 反向传播
    │
    ▼
predict.py ─── 自回归逐词生成翻译结果
    │
    ▼
evaluate.py ─── BLEU 评分
```

**关键超参数**（config.py）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `EMBEDDING_DIM` | 128 | 词向量维度 |
| `HIDDEN_SIZE` | 256 | GRU 隐藏层维度 |
| `BATCH_SIZE` | 64 | 每批训练样本数 |
| `MAX_SEQ_LEN` | 128 | 最大序列长度 |
| `LEARNING_RATE` | 1e-3 | 学习率 |
| `EPOCHS` | 10 | 训练轮数 |

---

## 2. 项目结构

```
translation-seq2seq/
├── data/
│   ├── raw/
│   │   └── cmn.txt              # 原始中英平行语料（tab分隔）
│   └── processed/
│       ├── train.jsonl           # 处理后的训练集（token_id序列）
│       └── test.jsonl            # 处理后的测试集
├── models/
│   ├── zh_vocab.txt              # 中文词表
│   ├── en_vocab.txt              # 英文词表
│   └── model.pth                 # 训练好的模型权重
├── logs/                         # TensorBoard 日志
└── src/
    ├── config.py                 # 全局配置
    ├── tokenizer.py              # 分词器（中文按字/英文按词）
    ├── process.py                # 数据预处理流程
    ├── dataset.py                # PyTorch Dataset + DataLoader
    ├── model.py                  # Encoder / Decoder / TranslationModel
    ├── train.py                  # 训练循环
    ├── predict.py                # 交互式翻译推理
    └── evaluate.py               # BLEU 评估
```

---

## 3. 配置文件 — config.py

### 完整代码

```python
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

MAX_SEQ_LEN = 128
SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 10
```

### 解释

- **`Path(__file__).parent.parent`**：`__file__` 是当前文件路径（`src/config.py`），`.parent` 取父目录（`src/`），再 `.parent` 取项目根目录。这样无论从哪里执行脚本，路径都能正确解析。
- **`/` 运算符**：`pathlib.Path` 重载了 `/` 运算符，`ROOT_DIR / "models"` 等价于 `"项目根/models"`，比 `os.path.join` 更直观。
- **`HIDDEN_SIZE = 256`**：隐藏层维度通常比 embedding 维度（128）大，给模型更大的表示空间来融合上下文信息。

---

## 4. 分词器 — tokenizer.py

### 4.1 特殊 Token

```python
class BaseTokenizer:
    unk_token = "<unk>"    # 未知词
    pad_token = "<pad>"    # 填充（对齐不同长度序列）
    sos_token = "<sos>"    # 句子开始 (Start Of Sentence)
    eos_token = "<eos>"    # 句子结束 (End Of Sentence)
```

> **为什么要这些特殊 token？**
> - `<pad>`：一个 batch 里的句子长度不同，短的用 `<pad>` 补到一样长
> - `<sos>` / `<eos>`：告诉 Decoder 何时开始生成、何时停止
> - `<unk>`：遇到词表里没有的词，统一映射为 `<unk>`

### 4.2 词表构建

```python
@classmethod
def build_vocab(cls, sentences, vocab_path):
    vocab_set = set()
    for sentence in tqdm(sentences, desc="构建词表"):
        vocab_set.update(cls.tokenize(sentence))

    vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] \
               + [token for token in vocab_set if token != ""]

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_list))
```

> **解释**：遍历所有句子，用 `set` 去重收集所有 token。**四个特殊 token 放在词表最前面**，所以它们的索引固定为 `0, 1, 2, 3`。

### 4.3 编码（文本 → ID 序列）

```python
def encode(self, text, add_sos_eos=False):
    tokens = self.tokenize(text)
    if add_sos_eos:
        tokens = [self.sos_token] + tokens + [self.eos_token]
    return [self.word2index.get(token, self.unk_token_index) for token in tokens]
```

> **解释**：`word2index.get(token, self.unk_token_index)` 是安全查字典——如果 token 不在词表中，返回 `<unk>` 的索引而不是报错。

### 4.4 中文 vs 英文分词

```python
class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)              # 中文按字符切："你好" → ["你", "好"]

class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)   # 英文按词切："hello world" → ["hello", "world"]

    def decode(self, indexes) -> str:
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)
```

> **关键差异**：
> - **中文**按字符切分（`list(text)`），因为中文词之间没有天然空格，直接用 jieba 分词也行但这里选用字符级更简单
> - **英文**用 NLTK 的 `TreebankWordTokenizer` 按词切分，能正确处理标点（如 `don't` → `do` + `n't`）
> - **decode** 只给英文写了，因为最终输出是英文，需要把 token ID 还原成可读文本

---

## 5. 数据预处理 — process.py

### 完整流程

```python
def process():
    # 1. 读取原始平行语料
    df = pd.read_csv(config.RAW_DATA_DIR/'cmn.txt', sep='\t',
                     header=None, usecols=[0, 1], names=['en', 'zh']).dropna()

    # 2. 划分训练集/测试集 (8:2)
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 3. 在训练集上构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.MODEL_DIR/'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.MODEL_DIR/'en_vocab.txt')

    # 4. 加载词表，创建分词器实例
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR/'en_vocab.txt')

    # 5. 将文本全部转为 token ID 序列
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 6. 保存为 JSONL
    train_df.to_json(config.PROCESSED_DATA_DIR/'train.jsonl', orient='records', lines=True)
    test_df.to_json(config.PROCESSED_DATA_DIR/'test.jsonl', orient='records', lines=True)
```

### 关键细节

> **为什么中文不加 `<sos>/<eos>`，英文要加？**
> - 中文是 Encoder 的输入，Encoder 只需要读到完整句子并输出一个上下文向量，不需要起止标记
> - 英文是 Decoder 的目标输出，训练时 Decoder 需要知道"从 `<sos>` 开始，到 `<eos>` 结束"

> **为什么要先划分数据集再建词表？**
> 词表只能从训练集构建——测试集模拟的是"没见过的新数据"，如果测试集参与了词表构建，就是数据泄露（data leakage）。

---

## 6. 数据集与加载器 — dataset.py

### 6.1 Dataset 类

```python
class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, orient="records", lines=True).to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]["zh"], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]["en"], dtype=torch.long)
        return input_tensor, target_tensor
```

> **三个必须实现的方法**：
> - `__init__`：加载数据
> - `__len__`：告诉 DataLoader 一共有多少条数据
> - `__getitem__`：按索引取一条数据，返回 `(中文ID序列, 英文ID序列)`

### 6.2 collate_fn — 批次拼接的关键

```python
def collate_fn(batch):
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]

    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_tensors, batch_first=True, padding_value=0)
    return input_tensor, target_tensor
```

> **为什么需要 `collate_fn`？**
>
> 一个 batch 里有 64 条句子，长短不一。PyTorch 默认要求一个 batch 内所有样本等长。`pad_sequence` 把短的句子用 `<pad>`（id=0）填充到和最长的句子一样长。
>
> ```
> 填充前: [[1,2,3], [4,5]]           → 长度 3 和 2
> 填充后: [[1,2,3], [4,5,0]]         → 都变成长度 3
> ```
>
> **`batch_first=True`**：让输出形状为 `[batch_size, seq_len]` 而非默认的 `[seq_len, batch_size]`。

### 6.3 DataLoader 工厂函数

```python
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset = TranslationDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
```

> 训练集 `shuffle=True` 打乱顺序，测试集不需要打乱。

---

## 7. 模型定义 — model.py

### 7.1 Encoder（编码器）

```python
class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx      # <pad> 对应的向量始终为 0，不参与梯度更新
        )
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )

    def forward(self, x):
        # x: [batch_size, src_seq_len]
        embed = self.embedding(x)
        # embed: [batch_size, src_seq_len, EMBEDDING_DIM]
        output, _ = self.gru(embed)
        # output: [batch_size, src_seq_len, HIDDEN_SIZE]

        # 取每个句子最后一个非 <pad> token 的隐藏状态
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]
        # last_hidden_state: [batch_size, HIDDEN_SIZE]
        return last_hidden_state
```

#### 计算图解

```
输入 x:  [batch_size, seq_len]          示例: [[5, 3, 7, 0, 0]]  (0 是 <pad>)
                │
                ▼
Embedding: [batch_size, seq_len, 128]   每个 token ID 变成一个 128 维向量
                │
                ▼
GRU:       [batch_size, seq_len, 256]   每个位置输出一个 256 维隐藏状态
                │
                ▼
取最后一个有效位置: [batch_size, 256]    跳过 padding，取真正"最后一个词"的状态
                                          这就是整个句子压缩后的"含义向量"
```

> **核心操作解释**：`output[torch.arange(output.shape[0]), lengths - 1]`
>
> ```python
> # 假设 batch_size=3, 3个句子的长度分别是 [5, 3, 7]
> torch.arange(3)  # → [0, 1, 2]   (batch 索引)
> lengths - 1      # → [4, 2, 6]   (每个句子最后一个有效位置的索引)
> # 组合起来就是取 output[0,4], output[1,2], output[2,6]
> ```

### 7.2 Decoder（解码器）

```python
class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx
        )
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )
        self.linear = nn.Linear(config.HIDDEN_SIZE, vocab_size)

    def forward(self, x, hidden_0):
        # x:        [batch_size, 1]          每次只输入 1 个 token
        # hidden_0: [1, batch_size, 256]     从 Encoder 传来的初始隐藏状态
        embed = self.embedding(x)
        # embed:    [batch_size, 1, 128]
        output, hidden_n = self.gru(embed, hidden_0)
        # output:   [batch_size, 1, 256]     当前步的隐藏状态
        # hidden_n: [1, batch_size, 256]     传给下一步的隐藏状态
        logits = self.linear(output)
        # logits:   [batch_size, 1, vocab_size]  每个词的"得分"
        return logits, hidden_n
```

#### 逐步生成示意图

```
第一步: 输入 <sos> + Encoder的context → GRU → 输出 "我"
第二步: 输入 "我"   + 上一步的hidden   → GRU → 输出 "喜欢"
第三步: 输入 "喜欢" + 上一步的hidden   → GRU → 输出 "编程"
...
第N步: 输出 <eos> → 停止
```

> **为什么 Decoder 的输入是 `[batch_size, 1]` 而不是整个序列？**
>
> 这是自回归（autoregressive）生成的核心：每次用上一步的输出作为下一步的输入。训练时用 Teacher Forcing（见下文），推理时用自己生成的 token。

### 7.3 组合模型

```python
class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size, padding_idx=zh_padding_idx)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size, padding_idx=en_padding_idx)
```

> **为什么要把 Encoder 和 Decoder 分开定义再组合？**
>
> 解耦（decoupling）：Encoder 只负责"理解中文"，Decoder 只负责"生成英文"。后续如果想换更强的 Encoder（比如 Transformer）或 Decoder，只需替换一个组件。

---

## 8. 训练脚本 — train.py

### 8.1 Teacher Forcing 策略

```python
encoder_inputs, targets = inputs.to(device), targets.to(device)
decoder_inputs = targets[:, :-1]    # 去掉最后一个 token
decoder_targets = targets[:, 1:]    # 去掉第一个 token
```

> **什么是 Teacher Forcing？**
>
> ```
> 原始目标序列: [<sos>, I, love, coding, <eos>]
> decoder_inputs:  [<sos>, I, love, coding]    ← 去掉最后的 <eos>
> decoder_targets: [I, love, coding, <eos>]    ← 去掉最前的 <sos>
> ```
>
> 在训练时，Decoder 每一步的输入**不是**自己上一步的预测，而是**标准答案**（ground truth）。这大大加速了训练收敛，因为不会让早期错误逐步累积。

### 8.2 训练循环核心

```python
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):
        # ---- 1. 数据准备 ----
        encoder_inputs, targets = inputs.to(device), targets.to(device)
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]

        # ---- 2. 编码 ----
        context_vector = model.encoder(encoder_inputs)
        # [batch_size, hidden_size]

        decoder_hidden = context_vector.unsqueeze(0)
        # [1, batch_size, hidden_size]  ← GRU 要求 3 维输入

        # ---- 3. 逐步解码 ----
        decoder_outputs = []
        seq_len = decoder_targets.shape[1]
        for i in range(seq_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)   # [batch_size, 1]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

        # ---- 4. 拼接所有时间步的输出 ----
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # [batch_size, seq_len, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # [batch_size * seq_len, vocab_size]

        decoder_targets = decoder_targets.reshape(-1)
        # [batch_size * seq_len]

        # ---- 5. 损失计算 + 反向传播 ----
        loss = loss_fn(decoder_outputs, decoder_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

> **`unsqueeze(0)` 的含义**：
>
> Encoder 输出是 `[batch, 256]`（2D），但 GRU 要求隐藏状态是 `[num_layers, batch, hidden]`（3D）。`unsqueeze(0)` 在第 0 维前面加一维：`[batch, 256]` → `[1, batch, 256]`。
> 这个 `1` 代表 1 层 × 1 方向 = 1。

> **`reshape(-1, ...)` 的含义**：
>
> CrossEntropyLoss 要求输入是 `[样本数, 类别数]`，标签是 `[样本数]`。
> 把 `[batch, seq_len, vocab_size]` 展平成 `[batch×seq_len, vocab_size]`，把所有时间步的所有样本合并成一个大矩阵，一次性算损失。

### 8.3 训练主函数

```python
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR/'en_vocab.txt')

    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_idx=zh_tokenizer.pad_token_index,
        en_padding_idx=en_tokenizer.pad_token_index
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float("inf")
    for epoch in range(1, 1 + config.EPOCHS):
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        writer.add_scalar("Loss", loss, epoch)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR/'model.pth')
```

> **`ignore_index=en_tokenizer.pad_token_index`**：计算损失时忽略 `<pad>` 位置——padding 本身没有意义，不应该参与损失计算，否则模型会浪费能力去"预测" padding。

> **TensorBoard 可视化**：用 `tensorboard --logdir=logs/` 启动，可以看到 Loss 曲线随 epoch 下降。

---

## 9. 推理/预测 — predict.py

### 9.1 批量预测（自回归生成）

```python
def predict_batch(model, inputs, en_tokenizer, device):
    model.eval()
    with torch.no_grad():                    # 推理不需要计算梯度
        context_vector = model.encoder(inputs)
        batch_size = inputs.shape[0]
        decoder_hidden = context_vector.unsqueeze(0)

        decoder_iutput = torch.full(
            size=(batch_size, 1),
            fill_value=en_tokenizer.sos_token_index,
            device=device
        )                                    # 起始符 <sos>

        generated = []
        is_finished = torch.full(size=(batch_size,), fill_value=False, device=device)

        for i in range(config.MAX_SEQ_LEN):
            decoder_output, decoder_hidden = model.decoder(decoder_iutput, decoder_hidden)
            next_token_indexes = decoder_output.argmax(dim=-1)   # 贪心解码：取概率最大的词
            generated.append(next_token_indexes)
            decoder_iutput = next_token_indexes                   # 用自己生成的 token 作为下一步输入
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():                                 # 所有句子都生成了 <eos>
                break

        generated_tensor = torch.cat(generated, dim=-1)           # [batch_size, seq_len]
        generated_list = generated_tensor.tolist()

        # 截断 <eos> 之后的内容
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        return generated_list
```

> **训练 vs 推理的关键区别**：
>
> | | 训练 (Teacher Forcing) | 推理 (自回归) |
> |---|---|---|
> | Decoder 输入 | 标准答案的 token | 自己上一步生成的 token |
> | 循环次数 | 已知序列长度 | 最多 `MAX_SEQ_LEN` 次，遇到 `<eos>` 提前停 |
> | 梯度 | 需要反向传播 | `torch.no_grad()` 关闭梯度 |

> **`argmax(dim=-1)`**：贪心解码（Greedy Decoding），每一步取概率最大的词。简单高效，但容易陷入局部最优。更高级的策略如 Beam Search 会保留多个候选。

### 9.2 交互式翻译

```python
def predict(text, model, zh_tokenizer, en_tokenizer, device):
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)
    batch_result = predict_batch(model, input_tensor, en_tokenizer, device)
    return en_tokenizer.decode(batch_result[0])

def run_predict():
    # ...加载资源...
    while True:
        user_input = input("中文: ")
        if user_input in ("q", "quit"):
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("翻译结果：", result)
```

---

## 10. 评估脚本 — evaluate.py

```python
def evaluate(model, test_dataloader, device, en_tokenizer):
    model.eval()
    predictions = []
    references = []

    for inputs, targets in tqdm(test_dataloader, desc="评估"):
        inputs = inputs.to(device)
        targets = targets.tolist()

        batch_results = predict_batch(model, inputs, en_tokenizer, device)

        predictions.extend(batch_results)
        references.extend([
            [target[1:target.index(en_tokenizer.eos_token_index)]]
            for target in targets
        ])

    bleu = corpus_bleu(references, predictions)
    return bleu
```

> **BLEU 评分**：对比机器翻译结果与人工参考译文的重叠程度（n-gram 匹配率），满分是 1.0（100% 匹配）。一般在 0.2~0.4 就算不错的结果。
>
> **`references` 的格式**：BLEU 允许每个源句有多个参考译文，所以 references 是三层嵌套：`[[[ref1]], [[ref2]], ...]`，即使只有一个参考也要包两层 `[[]]`。

---

## 11. 核心概念速查

### 11.1 Seq2Seq 架构

```
中文输入: "我 喜欢 编程"
              │
              ▼
         [Embedding]
              │
              ▼
         [Encoder GRU]  ─── 逐词读入，最终输出一个 context vector
              │
              │  context vector (整个中文句子的压缩表示)
              ▼
         [Decoder GRU]  ─── 逐词生成英文
              │
              ▼
英文输出: "I love coding"
```

### 11.2 张量形状变化一览

| 阶段 | 张量名称 | 形状 | 说明 |
|------|---------|------|------|
| 输入 | `x` | `[B, S_zh]` | B=batch, S_zh=中文序列长度 |
| Embedding | `embed` | `[B, S_zh, 128]` | 每个字变成一个 128 维向量 |
| Encoder 输出 | `last_hidden` | `[B, 256]` | 整句编码为一个向量 |
| Decoder hidden | `hidden` | `[1, B, 256]` | unsqueeze 适配 GRU 格式 |
| Decoder 输入 | `decoder_input` | `[B, 1]` | 每次只输入一个 token |
| Decoder 输出 | `logits` | `[B, 1, V_en]` | V_en=英文词表大小 |
| 全部输出 | `decoder_outputs` | `[B, S_en, V_en]` | S_en=英文序列长度 |

### 11.3 常见误区

1. **为什么中文不用 `<sos>/<eos>`？**
   中文走的是 Encoder，Encoder 只需要"读完整句 → 输出向量"，不需要知道起止。

2. **`padding_idx` 的作用？**
   `nn.Embedding(padding_idx=0)` 让 `<pad>` 的 embedding 始终为零向量，不参与梯度更新，避免 padding 干扰训练。

3. **`ignore_index` 的作用？**
   CrossEntropyLoss 计算时跳过 `<pad>` 位置，这样模型不会因为"正确预测了 padding"而获得虚假的低 loss。

4. **为什么损失函数输入要 reshape？**
   把所有 `(batch, seq_len)` 的 token 平铺成一个大列表，跟分类任务一样：每个 token 是一个"样本"，去预测它属于词表中哪个词。

5. **`squeeze` vs `squeeze_`？**
   带下划线 `_` 的是 in-place 操作（原地修改），如果多个变量引用同一 tensor 会导致连锁反应。推理时用 `squeeze(1)` 而不是 `squeeze_(1)`。
