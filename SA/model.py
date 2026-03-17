import torch # 导入 PyTorch 库
import torch.nn as nn # 导入神经网络模块
import torch.optim as optim # 导入优化器模块
import os
from collections import defaultdict
import math
from torch.nn.utils.rnn import pad_sequence

from tqdm import *
from torch.utils.tensorboard import SummaryWriter

class PositionalEncoding(nn.Module):
    """
    实现Transformer中使用的正弦位置编码。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe) # 注册为缓冲区，不作为模型参数训练

    def forward(self, x):
        """
        Args:
            x: 输入序列的嵌入，形状 (seq_len, batch_size, d_model)
        """
        # 将位置编码添加到输入嵌入中
        # 乘以sqrt(d_model) 是为了缩放嵌入，这在原始Transformer论文中是推荐的
        # 因为位置编码是在嵌入之后添加的，而嵌入的方差可能会导致其在加入位置编码后失去其相对值。
        # 乘以sqrt(d_model) 可以帮助保持嵌入的原始缩放。
        x = x * math.sqrt(x.size(-1)) 
        return x + self.pe[:x.size(0), :]
    
class TransformerLanguageModel(nn.Module):
    """
    一个基于Transformer Encoder的语言模型，用于下一个词元预测。
    使用两个Transformer Encoder层。
    """
    def set_vocabulary(self, vocabulary):
        special_tokens = {"<S>", "\n", "<UNK>"}
        self.vocabulary = sorted(list(vocabulary.union(special_tokens)))
    def reset_vocabulary(self, vocabulary):
        tmp_size = self.vocab_size
        self.set_vocabulary(vocabulary)
        assert len(self.vocabulary) == tmp_size, "New vocabulary size must match the original size."
        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocabulary)}

        self.start_token_id = self.token_to_id["<S>"]
        self.end_token_id = self.token_to_id["\n"]
        self.unknown_token_id = self.token_to_id["<UNK>"]
    def __init__(self, vocabulary, device, d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.set_vocabulary(vocabulary)
        
        self.start_token = "<S>"
        self.end_token = "\n"
        self.unknown_token = "<UNK>"

        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocabulary)}

        self.start_token_id = self.token_to_id["<S>"]
        self.end_token_id = self.token_to_id["\n"]
        self.unknown_token_id = self.token_to_id["<UNK>"]
        
        self.vocab_size = len(self.vocabulary)
        self.device = device
        self.d_model = d_model # Embedding dimension, also model dimension

        # 嵌入层
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.unknown_token_id)
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model, max_len=50) # Max sequence length assumed to be 50 for this dataset

        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 最终的线性层，将Transformer输出映射回词汇表大小
        self.output_linear = nn.Linear(d_model, self.vocab_size)
        # self.output_linear.weight = self.embedding.weight

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.lr = 0.0005
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)

    def forward(self, src):
        """
        前向传播。
        src: 输入序列的词元ID，形状 (seq_len, batch_size)
        """
        # 确保输入张量在正确的设备上
        src = src.to(self.device)

        # 1. 嵌入
        embedded = self.embedding(src) # (seq_len, batch_size, d_model)

        # 2. 添加位置编码
        embedded = self.pos_encoder(embedded) # (seq_len, batch_size, d_model)

        # 3. Transformer Encoder
        # TransformerEncoder 需要一个 src_mask 来防止在预测时看到未来的词元
        # 对于下一个词元预测，我们需要一个因果（causal）掩码，即一个上三角矩阵
        seq_len = src.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        output = self.transformer_encoder(embedded, mask=mask) # (seq_len, batch_size, d_model)

        # 4. 线性输出层
        # 我们希望预测每个位置的下一个词元，所以对所有位置都进行输出预测
        output = self.output_linear(output) # (seq_len, batch_size, vocab_size)
        return output

    def get_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

    def fit(self, tokenized_sequences, val_sequences, epochs=10, offset=0, epoch_offset=0, writer=None):
        processed_data = []
        tokenized_data = []
        for seq in tokenized_sequences:
            full_seq = [self.start_token] + [t if t != "\n" else self.end_token for t in seq] + [self.end_token]
            id_seq = [self.token_to_id.get(token, self.token_to_id[self.unknown_token]) for token in full_seq]
            tokenized_data.append(full_seq)
            processed_data.append(id_seq)
        val_data = []
        tokenized_val = []
        for seq in val_sequences:
            full_seq = [self.start_token] + [t if t != "\n" else self.end_token for t in seq] + [self.end_token]
            id_seq = [self.token_to_id.get(token, self.token_to_id[self.unknown_token]) for token in full_seq]
            tokenized_val.append(full_seq)
            val_data.append(id_seq)
        # print(self.token_to_id[self.end_token])
        # print(self.token_to_id[self.unknown_token])
        # print(processed_data[0])
        
        def get_batches(processed_data, batch_size):
            batches = []
            datalen = len(processed_data)
            for i in range(0, datalen, batch_size):
                r = min(i + batch_size, datalen)
                seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in processed_data[i:r]]
                seq_tensors = pad_sequence(seq_tensors, batch_first=False, padding_value=self.unknown_token_id)
                input_tensors = seq_tensors[:-1, :].clone()
                label_tensors = seq_tensors[1:, :].clone()
                label_tensors[label_tensors == self.unknown_token_id] = -100
                batches.append((input_tensors, label_tensors))
            return batches
        device = self.device
        
        self.train() # Ensure model is in training mode
        batches = get_batches(processed_data, 4096)
        val_batches = get_batches(val_data, 4096)
        global_step = offset
        global_epoch = epoch_offset
        for epoch in tqdm(range(epochs)):
            global_epoch += 1
            total_loss = 0
            num_samples = 0

            self.train()
            for input_ids, label_ids in batches:
                input_ids.to(device)
                self.optimizer.zero_grad()
                output = self.forward(input_ids)
                output = output.contiguous()
                label_ids = label_ids.contiguous()
                loss = self.loss_fn(output.view(-1, output.size(-1)).to(device), label_ids.view(-1).to(device))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_samples += input_ids.shape[1]

                writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step += 1
            if global_epoch % 5 == 0:
                losses = []
                self.eval()
                for input_ids, label_ids in val_batches:
                    input_ids.to(device)
                    with torch.no_grad():
                        output = self.forward(input_ids)
                    output = output.contiguous()
                    label_ids = label_ids.contiguous()
                    loss = self.loss_fn(output.view(-1, output.size(-1)).to(device), label_ids.view(-1).to(device))
                    
                    losses.append(loss.item())

                writer.add_scalar('Loss/val', sum(losses) / len(losses), global_epoch)
            if global_epoch % 40 == 0:
                acc = 0
                cnt = 0
                for line in tokenized_data[:100]:
                    eq_idx = line.index('=')
                    gold_answer = ''.join(line[eq_idx + 1:-1])
                    pred_answer = self.generate_sequence(line[1:eq_idx + 1])
                    if cnt < 10:
                        print(''.join(line[1:eq_idx + 1]),gold_answer, pred_answer)
                        cnt += 1
                    if pred_answer == gold_answer:
                        acc += 1
                writer.add_scalar('Acc/train', acc / 100, global_epoch)
                print(f"Epoch {global_epoch}, Acc: {acc / 100:.4f}")
                acc = 0
                for line in tokenized_val:
                    eq_idx = line.index('=')
                    gold_answer = ''.join(line[eq_idx + 1:-1])
                    pred_answer = self.generate_sequence(line[1:eq_idx + 1])
                    if pred_answer == gold_answer:
                        acc += 1
                writer.add_scalar('Acc/val', acc / len(tokenized_val), global_epoch)

            if num_samples > 0:
                avg_loss = total_loss / num_samples
                print(f"Epoch {global_epoch}, Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {global_epoch}, No training samples.")
        return global_step, global_epoch


    def predict_next_token(self, context_tokens_list):
        processed_context = [self.start_token] + context_tokens_list
        input_ids = [self.token_to_id.get(token, self.token_to_id[self.unknown_token]) for token in processed_context]
        # print(input_ids)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(self.device)

        self.eval() 
        with torch.no_grad(): 
            output = self.forward(input_tensor) 
            last_token_output = output[-1, 0, :]
            probabilities = torch.softmax(last_token_output, dim=-1) 
            predicted_id = torch.argmax(probabilities).item()
        self.train() 

        return self.id_to_token[predicted_id]

    def generate_sequence(self, temp_tokenized, max_gen_length=15):
        predicted_sequence = list(temp_tokenized)

        full_generated_text = ""
        for _ in range(max_gen_length):
            next_token = self.predict_next_token(predicted_sequence) 
            if next_token == self.end_token:
                # print(f"预测到序列结束标记，停止生成。")
                break
            predicted_sequence.append(next_token)
            full_generated_text += next_token
            # print(f"当前生成: '{full_generated_text}'")

        # print(f"\n最终预测序列: '{''.join(predicted_sequence)}'")
        return full_generated_text