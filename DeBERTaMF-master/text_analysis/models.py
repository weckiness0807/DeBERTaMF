'''
Created on Feb, 2024

@author: Wen Yao
'''
import numpy as np
np.random.seed(1337)

from transformers import DebertaModel, DebertaTokenizer
import torch
from torch import nn

class DeBERTa_Model(torch.nn.Module):
    def __init__(self, output_dimesion, hidden_dimension = 64, num_layers = 1,learning_rate=0.001):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.rnn = nn.GRU(self.deberta.config.hidden_size, hidden_dimension, num_layers, batch_first=True)
        self.feature_projector = torch.nn.Linear(hidden_dimension, output_dimesion)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        rnn_outputs, _ = self.rnn(outputs.last_hidden_state)
        # 取最后一个时间步的输出
        last_rnn_output = rnn_outputs[:, -1, :]
        logits = self.feature_projector(last_rnn_output)
        return logits

    def train(self, inputs, labels, item_weight, learning_rate=0.001, batch_size=32):
        self.deberta.train()
        total_loss = 0
        # 将输入分解成多个批次
        for i in range(0, len(inputs['input_ids']), batch_size):
            batch_inputs = {
                'input_ids': inputs['input_ids'][i:i+batch_size],
                'attention_mask': inputs['attention_mask'][i:i+batch_size],
            }
            batch_labels = torch.from_numpy(labels[i:i+batch_size])
            batch_item_weight = torch.from_numpy(item_weight[i:i+batch_size])
            self.optimizer.zero_grad()

            outputs = self.deberta(**batch_inputs).last_hidden_state
            outputs, _ = self.rnn(outputs)  # 使用GRU处理outputs
            outputs = outputs[:, -1, :]
            logits = self.feature_projector(outputs)
            
            loss_fn = torch.nn.MSELoss(reduction='none')  # 不立即求和
            loss = loss_fn(logits, batch_labels)
            batch_item_weight = batch_item_weight.unsqueeze(1).expand_as(loss)
            loss = (loss * batch_item_weight).mean()  # 加权平均
            loss.backward()  # 在每个批次处理完之后进行一次反向传播
            self.optimizer.step()  # 在每个批次处理完之后进行一次参数更新
            total_loss += loss.item()
        return total_loss / (len(inputs['input_ids']) // batch_size)

    def extract_features(self, inputs,batch_size=100):
        self.deberta.eval()
        features = []
        with torch.no_grad():
            for i in range(0, len(inputs['input_ids']), batch_size):
                batch_inputs = {
                'input_ids': inputs['input_ids'][i:i+batch_size],
                'attention_mask': inputs['attention_mask'][i:i+batch_size],
                }

                deberta_outputs = self.deberta(**batch_inputs)
                rnn_outputs, _ = self.rnn(deberta_outputs.last_hidden_state)
                last_rnn_output = rnn_outputs[:, -1, :]
                logits = self.feature_projector(last_rnn_output)

                logits = logits.detach().numpy() 
                features.extend(logits)
        return np.array(features)

