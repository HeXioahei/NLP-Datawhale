# （一）在数据集上进行预训练

## 一、需要导入的包

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
import random
from torch.utils.data import Subset, DataLoader
import time
```

## 二、定义数据集类

1. 从文件中读取数据并存储为元组列表。

2. 创建词汇表，确保术语词典中的词被包含在内。

3. 使用分词器将句子分词并转换为索引张量。

4. 提供获取数据集长度和根据索引获取数据的方法。

（代码里有详细注解）

**附**：[什么是Counter](https://blog.csdn.net/qq_41813454/article/details/136963165?ops_request_misc=&request_id=&biz_id=102&utm_term=Counter&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-136963165.nonecase&spm=1018.2226.3001.4187)、[Counter的常用方法](https://blog.csdn.net/chl183/article/details/106956807?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172094207416800225539107%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172094207416800225539107&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-7-106956807-null-null.142^v100^pc_search_result_base9&utm_term=Counter&spm=1018.2226.3001.4187)

*张量是深度学习和科学计算中广泛使用的一种数据结构，可以视为多维数组或矩阵的推广。在PyTorch、TensorFlow等深度学习框架中，张量是进行各种计算和操作的基本单元。张量具有维度（Dimension）的概念，可以是0维（标量）、1维（向量）、2维（矩阵）或更高维度的数组。*

```python
class TranslationDataset(Dataset):               #  TranslationDataset 类继承自 Dataset 类。
    def __init__(self, filename, terminology):   # 传入文件名 filename 和术语词典 terminology 。
        self.data = []
        # 打开文件并逐行读取，每行按制表符`\t`分割成英文和中文，然后将它们作为元组添加到`self.data`列表中。
        with open(filename, 'r', encoding='utf-8') as f:     
            for line in f:
                en, zh = line.strip().split('\t')
                self.data.append((en, zh))

        self.terminology = terminology

        # 创建分词器
        self.en_tokenizer = get_tokenizer('basic_english')  # 使用 get_tokenizer('basic_english') 获取英文分词器。
        self.zh_tokenizer = list  # 使用`list`作为**中文分词器**，表示按**字符级**分词。

        # 创建词汇表，注意这里需要确保术语词典中的词也被包含在词汇表中
        # 初始化英文和中文词汇表为 Counter对象
        en_vocab = Counter(self.terminology.keys())  #  将术语词典的键添加到英文词汇表中
        zh_vocab = Counter()

        # 遍历数据集中的每一对英文和中文句子，使用分词器更新词汇表。
        for en, zh in self.data:
            en_vocab.update(self.en_tokenizer(en))
            zh_vocab.update(self.zh_tokenizer(zh))

        # 添加术语到词汇表
        self.en_vocab = ['<pad>', '<sos>', '<eos>'] + list(self.terminology.keys()) + [word for word, _ in en_vocab.most_common(10000)]  # 将特殊标记`<pad>`, `<sos>`, `<eos>`和术语词典的键添加到英文词汇表中，然后添加出现频率最高的10000个词。
        self.zh_vocab = ['<pad>', '<sos>', '<eos>'] + [word for word, _ in zh_vocab.most_common(10000)]

        # 创建词到索引的映射 en_word2idx 和 zh_word2idx 。
        self.en_word2idx = {word: idx for idx, word in enumerate(self.en_vocab)}
        self.zh_word2idx = {word: idx for idx, word in enumerate(self.zh_vocab)}

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 根据索引获取数据，将英文和中文句子分词并转换为索引张量，最后返回这些张量
    def __getitem__(self, idx):
        en, zh = self.data[idx]
        en_tensor = torch.tensor([self.en_word2idx.get(word, self.en_word2idx['<sos>']) for word in self.en_tokenizer(en)] + [self.en_word2idx['<eos>']])
        zh_tensor = torch.tensor([self.zh_word2idx.get(word, self.zh_word2idx['<sos>']) for word in self.zh_tokenizer(zh)] + [self.zh_word2idx['<eos>']])
        return en_tensor, zh_tensor
```

## 三、定义整理函数

这里定义了一个名为 `collate_fn` 的函数，它接受一个参数 `batch`（中文意思：批）。这个函数的主要目的是对一批数据进行处理，使其适合输入到<u>神经网络</u>中。

1. 初始化两个空列表用于存储英文和中文的序列数据。

2. 遍历批处理数据，将每对英文和中文的序列数据分别添加到对应的列表中。

3. 使用 `pad_sequence` 函数对两个列表中的序列进行填充，使其长度一致。

4. 返回填充后的英文和中文序列张量。

（代码里有详细注解）

```python
def collate_fn(batch):
    en_batch, zh_batch = [], []

    for en_item, zh_item in batch:
        en_batch.append(en_item)
        zh_batch.append(zh_item)

    # 对英文和中文序列分别进行填充
    en_batch = nn.utils.rnn.pad_sequence(en_batch, padding_value=0, batch_first=True)
    zh_batch = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0, batch_first=True)
    #  pad_sequence 函数会将序列列表中的所有序列填充到相同的长度，以便它们可以作为一个批次输入到神经网络中。padding_value=0 表示使用 0 进行填充，batch_first=True 表示输出的张量中批次维度在第一维。

    return en_batch, zh_batch
```

## 四、定义编码器

编码器的主要任务是将输入序列编码为一个固定维度的隐藏状态，这个隐藏状态将作为解码器的初始状态，用于生成目标序列。

其主要功能如下：

* **嵌入层**：将输入的整数索引序列转换为密集向量表示。
* **GRU层**：处理嵌入后的序列数据，生成每个时间步的输出和最后一个时间步的隐藏状态。
* **全连接层**：将GRU的输出转换为预测的词汇表概率分布。
* **Dropout层**：在训练过程中随机丢弃一些神经元，以防止过拟合。

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)  
        # self.embedding ：一个嵌入层，将输入的整数索引转换为密集向量。
        # input_dim 是词汇表的大小，emb_dim 是嵌入向量的维度。

        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)  
        # self.rnn ：一个 GRU 层，用于处理序列数据。
        # emb_dim 是输入维度，hid_dim 是隐藏状态的维度，
        # n_layers 是 GRU 的层数，dropout 是 dropout 率，
        # batch_first=True 表示输入和输出的第一个维度是 batch 大小。

        self.dropout = nn.Dropout(dropout)
        # self.dropout ：一个dropout层，用于防止过拟合。

    # 定义数据在模型中的流动过程
    def forward(self, src):
        # src 是输入的源序列，形状为 [batch_size, src_len]，其中 batch_size 是批量大小，src_len 是源序列的长度。

        embedded = self.dropout(self.embedding(src))
        # embedded 是通过嵌入层和 dropout 层处理后的结果，形状为 [batch_size, src_len, emb_dim]。

        outputs, hidden = self.rnn(embedded)
        # outputs 和 hidden 是通过 GRU 层处理后的结果。
        # outputs 包含每个时间步的输出，形状为 [batch_size, src_len, hid_dim]。
        # hidden 是最后一个时间步的隐藏状态，形状为 [n_layers, batch_size, hid_dim]。

        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim  # output_dim：输出维度，即目标词汇表的大小。
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)  # fc_out：全连接层，将GRU的输出转换为预测的词汇表概率分布。
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input 是当前时间步的输入，形状为 [batch_size, 1]。
        # hidden 是 GRU 的隐藏状态，形状为 [n_layers, batch_size, hid_dim]。

        embedded = self.dropout(self.embedding(input))
        # embedded 是通过嵌入层和 Dropout 层处理后的输入，形状为 [batch_size, 1, emb_dim]。

        output, hidden = self.rnn(embedded, hidden)
        # output 和 hidden 是通过 GRU 层处理后的输出和新的隐藏状态。
        # output 的形状为 [batch_size, 1, hid_dim]，hidden 的形状保持不变。

        prediction = self.fc_out(output.squeeze(1))
        # prediction 是通过全连接层将 GRU 的输出转换为预测的词汇表概率分布，形状为 [batch_size, output_dim]。

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src：源序列，形状为 [batch_size, src_len]。
        # trg：目标序列，形状为 [batch_size, trg_len]。
        # teacher_forcing_ratio：教师强制比率，默认为 0.5。

        # 初始化输出张量
        batch_size = src.shape[0]  # 获取批次大小 batch_size 
        trg_len = trg.shape[1]  # 获取目标序列长度 trg_len
        trg_vocab_size = self.decoder.output_dim  # 获取目标词汇表大小 trg_vocab_size

        # 初始化一个全零张量 outputs，形状为 [batch_size, trg_len, trg_vocab_size]，并将其移动到指定设备。
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码器处理源序列 src，得到编码器的输出 outputs 和隐藏状态 hidden。
        _, hidden = self.encoder(src)

        # 将目标序列的第一个词作为解码器的初始输入，形状为 [batch_size, 1]。
        input = trg[:, 0].unsqueeze(1)  # Start token

        # 循环解码
        for t in range(1, trg_len):  # 从第二个词开始，循环处理目标序列的每个词。

            output, hidden = self.decoder(input, hidden)  
            # 使用解码器处理当前输入 input 和隐藏状态 hidden，得到输出 output 和新的隐藏状态 hidden。

            outputs[:, t, :] = output  
            # 将 output 存储到 outputs 张量的相应位置。

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            # 根据教师强制比率决定下一个输入是使用目标序列的下一个词还是使用当前输出的最高概率词。

        return outputs
```

## 五、加载新增术语词典

从一个包含术语及其对应中文翻译的文件中读取数据，并将其存储在一个字典中。具体来说，它逐行读取文件内容，按制表符分割每行内容，然后将英文术语作为键、中文术语作为值存储在字典中。这个字典可以用于后续的术语翻译或其他处理。

```python
# 新增术语词典加载部分
def load_terminology_dictionary(dict_file):    # 这个参数是一个文件路径，指向包含术语词典的文件。
    terminology = {}                           # 初始化一个空字典，用于存储从文件中读取的术语及其对应的中文翻译。
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:                         # 逐行读取文件内容。每行代表一个术语及其对应的中文翻译。
            en_term, ch_term = line.strip().split('\t')   # 对于每一行，首先使用 strip() 方法去除行末的换行符和其他空白字符，然后使用 split('\t') 方法按制表符（'\t'）分割行内容，得到英文术语 en_term 和中文术语 ch_term。
            terminology[en_term] = ch_term                # 将英文术语 en_term 作为键，中文术语 ch_term 作为值，添加到 terminology 字典中。
    return terminology
```

## 六、训练神经网络模型

实现一个训练循环，用于训练一个神经网络模型，有效地更新模型参数以最小化损失函数。

具体步骤包括：

1. 设置模型为训练模式。
2. 遍历数据迭代器，获取批次数据并将其移动到指定设备。
3. 进行前向传播，计算损失。
4. 进行反向传播，更新模型参数，并对梯度进行裁剪。
5. 累积整个 epoch 的损失，并返回平均损失。

```python
def train(model, iterator, optimizer, criterion, clip):
    # 五个参数：model（模型）、iterator（数据迭代器）、optimizer（优化器）、criterion（损失函数）和 clip（梯度裁剪值）。

    model.train()                                   # 设置模型为训练模式
    epoch_loss = 0                                  # 初始化为零，用于累积整个 epoch 的损失。
    for i, (src, trg) in enumerate(iterator):       # 使用 enumerate(iterator) 遍历数据迭代器，每次迭代获取一个批次的数据 src 和 trg。
        src, trg = src.to(device), trg.to(device)   # 将 src 和 trg 移动到指定的设备（如 GPU）
        optimizer.zero_grad()                       # optimizer.zero_grad() 清空优化器的梯度，防止梯度累积。
        output = model(src, trg)                    # output = model(src, trg) 进行前向传播，得到模型的输出。
        output_dim = output.shape[-1]               # output_dim 是输出的最后一个维度的大小。
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)      # output 和 trg 都去掉第一个元素（通常是起始标记），然后重新排列形状以便于计算损失。
        loss = criterion(output, trg)               # 计算损失
        loss.backward()                             # 进行反向传播，计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # 对梯度进行裁剪，防止梯度爆炸。
        optimizer.step()                            # 更新模型参数。
        epoch_loss += loss.item()                   # 累积当前批次的损失。
    return epoch_loss / len(iterator)               # 返回整个 epoch 的平均损失
```

## 七、主函数

训练一个英中翻译模型，实现从数据加载到模型训练再到模型保存的全过程。

具体步骤包括：

1. 加载数据集和术语词典。
2. 选择数据集的子集进行训练。
3. 定义模型的参数和结构。
4. 初始化模型、优化器和损失函数。
5. 进行模型训练，并在每个 epoch 结束时打印训练损失。
6. 训练结束后保存模型权重。
7. 计算并打印程序的总运行时间。

```python
# 主函数
if __name__ == '__main__':
    start_time = time.time()  # 开始计时

    # 选择执行计算的设备，优先选择 GPU（如果可用），否则使用 CP。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # 加载英中术语词典，用于后续的数据处理。
    terminology = load_terminology_dictionary('../dataset/en-zh.dic')

    # 加载翻译数据集，并传入术语词典。
    dataset = TranslationDataset('../dataset/train.txt',terminology = terminology)

    # 选择数据集的前N个样本进行训练
    N = 1000  #int(len(dataset) * 1)  # 或者你可以设置为数据集大小的一定比例，如 int(len(dataset) * 0.1)
    subset_indices = list(range(N))
    subset_dataset = Subset(dataset, subset_indices)

    # 创建数据加载器，设置批量大小为 32，打乱数据顺序，并使用自定义的 collate_fn 函数处理数据
    train_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 定义模型的各项参数，包括输入维度、输出维度、嵌入维度、隐藏层维度、层数和 dropout 率。
    INPUT_DIM = len(dataset.en_vocab)    # 输入维度
    OUTPUT_DIM = len(dataset.zh_vocab)   # 输出维度
    ENC_EMB_DIM = 256                    # 嵌入维度
    DEC_EMB_DIM = 256                    # 嵌入维度
    HID_DIM = 512                        # 隐藏层维度
    N_LAYERS = 2                         # 层数
    ENC_DROPOUT = 0.5                    # dropout 率
    DEC_DROPOUT = 0.5                    # dropout 率

    # 初始化模型，初始化编码器和解码器，并组合成一个序列到序列（Seq2Seq）模型，然后将模型移动到指定设备。
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    # 定义优化器为 Adam，损失函数为交叉熵损失，并忽略填充符（<pad>）的损失。
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.zh_word2idx['<pad>'])

    # 设置训练的 epoch 数为 10，梯度裁剪值为 1
    N_EPOCHS = 10
    CLIP = 1

    # 进行训练，并在每个 epoch 结束时打印训练损失。
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')

    # 训练循环结束后，保存模型的权重
    torch.save(model.state_dict(), './translation_model_GRU.pth')

    end_time = time.time()  # 结束计时

    # 计算并打印运行时间
    elapsed_time_minute = (end_time - start_time)/60
    print(f"Total running time: {elapsed_time_minute:.2f} minutes")
```

# （二）在开发集上进行模型评价

## 一、引入需要的包

```python
import torch
from sacrebleu.metrics import BLEU
from typing import List
```

## 二、加载句子

从指定路径的文件中读取所有句子，并将它们存储在一个字符串列表中返回。

```python
# 假设我们已经定义了TranslationDataset, Encoder, Decoder, Seq2Seq类

# 这个函数的主要目的是从指定的文件中读取所有句子，并将它们存储在一个列表中返回。
# file_path 是输入参数，表示要读取的文件的路径。
# -> List[str] 是函数的返回类型注解，表示函数将返回一个字符串列表。
def load_sentences(file_path: str) -> List[str]:  
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]  # 使用列表推导式读取文件的每一行，去除前后空白字符，并将处理后的行内容存储在列表中返回
```

## 三、考虑术语词典

使用Seq2Seq模型将英语句子翻译成中文，并在翻译过程中考虑术语词典，以确保翻译结果中的术语准确。

具体步骤包括：

1. 将输入句子分词并转换为索引张量。
2. 使用编码器获取句子的隐藏状态。
3. 使用解码器逐步生成翻译结果，并在每一步检查生成的词是否在术语词典中，如果是则替换为术语词典中的对应词。
4. 返回最终的翻译结果字符串。

```python
# 更新translate_sentence函数以考虑术语词典
# sentence: 需要翻译的句子。model: 用于翻译的 Seq2Seq 模型。dataset: 包含词典和分词器的翻译数据集。
# terminology: 术语词典，用于替换翻译结果中的术语。device: 模型运行的设备（CPU或GPU）。max_length: 翻译结果的最大长度，默认为 50。
def translate_sentence(sentence: str, model: Seq2Seq, dataset: TranslationDataset, terminology, device: torch.device, max_length: int = 50):
    model.eval()       # 将模型设置为评估模式，关闭 dropout 和 batch normalization 等训练时使用的层。
    tokens = dataset.en_tokenizer(sentence)         # 使用数据集中的英语分词器对句子进行分词。
    tensor = torch.LongTensor([dataset.en_word2idx.get(token, dataset.en_word2idx['<sos>']) for token in tokens]).unsqueeze(0).to(device)  # [1, seq_len]  将分词结果转换为索引张量，并添加一个维度（batch维度），然后将张量移动到指定设备。

    with torch.no_grad():
        _, hidden = model.encoder(tensor)  # 关闭梯度计算，进行编码器的前向传播，获取隐藏状态。

    # 初始化翻译结果列表和解码器的输入标记
    translated_tokens = []
    input_token = torch.LongTensor([[dataset.zh_word2idx['<sos>']]]).to(device)  # [1, 1]

    # 循环进行解码器的前向传播，直到达到最大长度或遇到结束标记<eos>。
    for _ in range(max_length):
        output, hidden = model.decoder(input_token, hidden)
        top_token = output.argmax(1)
        translated_token = dataset.zh_vocab[top_token.item()]

        if translated_token == '<eos>':
            break

        # 如果翻译的词在术语词典中，则使用术语词典中的词
        if translated_token in terminology.values():
            for en_term, ch_term in terminology.items():
                if translated_token == ch_term:
                    translated_token = en_term
                    break

        # 将翻译结果添加到列表中，并更新输入标记
        translated_tokens.append(translated_token)
        input_token = top_token.unsqueeze(1)  # [1, 1]

    return ''.join(translated_tokens)     # 将翻译结果列表拼接成字符串并返回


```

## 四、评估翻译质量

量化评估一个序列到序列（Seq2Seq）模型的翻译质量。

具体步骤如下：

1. 加载源语言句子和参考翻译句子。
2. 使用模型逐句翻译源语言句子。
3. 计算翻译结果与参考翻译之间的 BLEU 分数。
4. 返回 BLEU 分数作为翻译质量的评估指标。

```python
# model: 一个 Seq2Seq 模型，用于翻译。dataset: 一个 TranslationDataset 数据集。src_file: 源语言句子文件的路径。
# ref_file: 参考翻译句子文件的路径。terminology: 术语信息。device: 使用的设备（如 CPU 或 GPU）。
def evaluate_bleu(model: Seq2Seq, dataset: TranslationDataset, src_file: str, ref_file: str, terminology,device: torch.device):
    model.eval()     # 将模型设置为评估模式，这意味着模型将不会进行训练，也不会更新权重

    # 使用 load_sentences 函数加载源语言句子和参考翻译句子。
    src_sentences = load_sentences(src_file)
    ref_sentences = load_sentences(ref_file)

    # 初始化一个空列表 translated_sentences，用于存储翻译后的句子
    translated_sentences = []

    for src in src_sentences:
        translated = translate_sentence(src, model, dataset, terminology, device)  # 翻译
        translated_sentences.append(translated)    # 再将翻译结果添加到 translated_sentences 列表中。

    # 计算翻译后的句子列表 translated_sentences 和参考翻译句子列表 ref_sentences 之间的 BLEU 分数。
    bleu = BLEU()
    score = bleu.corpus_score(translated_sentences, [ref_sentences])

    return score
```

## 五、主函数

加载一个预训练的Seq2Seq翻译模型，并使用该模型对开发集进行翻译，然后评估翻译结果的BLEU-4分数。

具体步骤包括：

1. 选择计算设备（GPU或CPU）。
2. 加载术语词典。
3. 创建翻译数据集实例。
4. 定义模型参数。
5. 初始化Seq2Seq模型。
6. 加载预训练的模型权重。
7. 使用模型对开发集进行翻译并计算BLEU-4分数。

```python
# 主函数
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载术语词典
    terminology = load_terminology_dictionary('../dataset/en-zh.dic')

    # 创建数据集实例时传递术语词典
    dataset = TranslationDataset('../dataset/train.txt', terminology)


    # 定义模型参数
    INPUT_DIM = len(dataset.en_vocab)
    OUTPUT_DIM = len(dataset.zh_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # 初始化模型
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load('./translation_model_GRU.pth'))

    # 评估BLEU分数
    bleu_score = evaluate_bleu(model, dataset, '../dataset/dev_en.txt', '../dataset/dev_zh.txt', terminology = terminology,device = device)
    print(f'BLEU-4 score: {bleu_score.score:.2f}')
```



# （三）在测试集上进行推理

## 一、定义模型接口

使用一个 `Seq2Seq` 模型对源文件中的句子进行翻译，并将翻译结果保存到一个新的文件中。

具体步骤如下：

1. 将模型设置为评估模式。
2. 加载源文件中的句子。
3. 逐句进行翻译，并将翻译结果存储在一个列表中。
4. 将翻译结果列表连接成一个字符串，每个句子之间用换行符分隔。
5. 将连接好的字符串写入到一个新的文件中。

```python
def inference(model: Seq2Seq, dataset: TranslationDataset, src_file: str, save_dir:str, terminology, device: torch.device):
    model.eval()
    src_sentences = load_sentences(src_file)

    translated_sentences = []
    for src in src_sentences:
        translated = translate_sentence(src, model, dataset, terminology, device)
        #print(translated)
        translated_sentences.append(translated)
        #print(translated_sentences)

    # 将列表元素连接成一个字符串，每个元素后换行
    text = '\n'.join(translated_sentences)

    # 打开一个文件，如果不存在则创建，'w'表示写模式
    with open(save_dir, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(text)

    #return translated_sentences
```

## 二、主函数

加载一个预训练的Seq2Seq翻译模型，并使用该模型对测试数据进行翻译。

具体步骤包括：

1. 选择运行模型的设备（GPU或CPU）。
2. 加载术语词典和翻译数据集。
3. 定义模型的各种参数。
4. 初始化编码器和解码器，并组合成Seq2Seq模型。
5. 加载预训练的模型权重。
6. 进行翻译推理，并将结果保存到指定文件。

```python
# 主函数
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载术语词典
    terminology = load_terminology_dictionary('../dataset/en-zh.dic')
    # 加载数据集和模型
    dataset = TranslationDataset('../dataset/train.txt',terminology = terminology)

    # 定义模型参数
    INPUT_DIM = len(dataset.en_vocab)
    OUTPUT_DIM = len(dataset.zh_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # 初始化模型
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load('./translation_model_GRU.pth'))

    save_dir = '../dataset/submit.txt'
    inference(model, dataset, src_file="../dataset/test_en.txt", save_dir = save_dir, terminology = terminology, device = device)
    print(f"翻译完成！文件已保存到{save_dir}")
```
