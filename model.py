import torch
import torch.nn as nn
import math

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_len=90):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


### -------- Transformer 模型 -------- ###
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_len=90):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_len = output_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, output_len)

    def forward(self, x):
        # x: (batch_size, input_len, input_dim)
        x = self.embedding(x)  # -> (batch, seq_len, d_model)
        x = self.positional_encoding(x)  # 加入位置编码
        x = self.transformer_encoder(x)  # -> (batch, seq_len, d_model)
        out = self.decoder(x[:, -1, :])  # 只用最后一个时间步的表示做预测
        return out


### -------- 位置编码（标准实现） -------- ###
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)




class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat([x1, x2, x3], dim=1)  # concat on channels

class InceptionTransformer(nn.Module):
    def __init__(self, input_dim, output_len=90, d_model=128, nhead=4):
        super(InceptionTransformer, self).__init__()
        self.inception = InceptionBlock(input_dim, out_channels=32)  # → output 96 channels
        self.linear_proj = nn.Linear(96, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, output_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.inception(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = self.linear_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.decoder(x[:, -1])



class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)

class TransformerWithLearnedPE(nn.Module):
    def __init__(self, input_dim, output_len=90, d_model=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = LearnablePositionalEncoding(max_len=500, d_model=d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.Linear(d_model, output_len)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.decoder(x[:, -1])


import torch
import torch.nn as nn


class MultiScaleConvAdaptiveTransformer(nn.Module):
    def __init__(self, input_dim, output_len, num_filters=64, d_model=128, num_heads=8, num_layers=2):
        super(MultiScaleConvAdaptiveTransformer, self).__init__()
        # 多尺度卷积层
        self.conv_short = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(input_dim, num_filters, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(input_dim, num_filters, kernel_size=15, padding=7)
        self.conv_proj = nn.Linear(num_filters * 3, d_model)  # 投影到Transformer维度

        # 自适应注意力层
        self.adaptive_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 生成注意力权重
        )

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]

        # 多尺度卷积
        short = self.conv_short(x)
        mid = self.conv_mid(x)
        long = self.conv_long(x)
        conv_out = torch.cat([short, mid, long], dim=1)  # [batch, num_filters*3, seq_len]
        conv_out = conv_out.permute(0, 2, 1)  # [batch, seq_len, num_filters*3]
        conv_out = self.conv_proj(conv_out)  # [batch, seq_len, d_model]

        # 自适应注意力
        attn_weights = self.adaptive_attention(conv_out)  # [batch, seq_len, d_model]
        x = conv_out * attn_weights  # 加权特征

        # Transformer编码
        x = self.transformer(x)  # [batch, seq_len, d_model]
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.fc(x)  # [batch, output_len]
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEnhancedTemporalModel(nn.Module):
    def __init__(self, input_dim, output_len, hidden_dim=64, num_nodes=13):  # num_nodes = 特征数
        super(GraphEnhancedTemporalModel, self).__init__()
        # 图嵌入层
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.adj = nn.Parameter(torch.ones(num_nodes, num_nodes) * 0.1)  # 可学习邻接矩阵

        # GNN层
        self.gnn = nn.Linear(hidden_dim * 2, hidden_dim)  # 聚合节点和邻居信息

        # TCN层
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim=num_nodes]
        batch_size, seq_len, num_nodes = x.shape

        # 图嵌入
        x_emb = self.node_embedding(x)  # [batch, seq_len, num_nodes, hidden_dim]
        adj = torch.softmax(self.adj, dim=1)  # 归一化邻接矩阵

        # GNN更新
        neighbor_info = torch.matmul(adj, x_emb)  # [batch, seq_len, num_nodes, hidden_dim]
        x_gnn = torch.cat([x_emb, neighbor_info], dim=-1)  # [batch, seq_len, num_nodes, hidden_dim*2]
        x_gnn = F.relu(self.gnn(x_gnn))  # [batch, seq_len, num_nodes, hidden_dim]

        # TCN处理时间维度
        x_tcn = x_gnn.permute(0, 2, 1, 3).reshape(batch_size, num_nodes * hidden_dim, seq_len)
        x_tcn = self.tcn(x_tcn)  # [batch, num_nodes*hidden_dim, seq_len]
        x_tcn = x_tcn.permute(0, 2, 1)[:, -1, :]  # [batch, num_nodes*hidden_dim]

        # 输出预测
        x = self.fc(x_tcn)  # [batch, output_len]
        return x


import torch
import torch.nn as nn


class DynamicAttentionExternalModel(nn.Module):
    def __init__(self, input_dim, external_dim, output_len, d_model=128, num_heads=8):
        super(DynamicAttentionExternalModel, self).__init__()
        # 分离编码器
        self.power_encoder = nn.Linear(input_dim - 5, d_model)  # 排除5个外部因素
        self.external_encoder = nn.Linear(external_dim, d_model)  # 外部因素维度=5

        # 动态注意力
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # 前馈网络与残差连接
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.fc = nn.Linear(d_model, output_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim], 其中最后5列为外部因素
        power_data = x[:, :, :-5]  # 电力相关数据
        external_data = x[:, :, -5:]  # 外部因素

        # 编码
        power_emb = self.power_encoder(power_data)  # [batch, seq_len, d_model]
        external_emb = self.external_encoder(external_data)  # [batch, seq_len, d_model]

        # 动态注意力融合
        attn_output, _ = self.attention(power_emb, external_emb, external_emb)
        x = power_emb + attn_output  # 残差连接

        # 前馈网络
        x = self.ffn(x[:, -1, :])  # 取最后一个时间步
        x = self.fc(x)  # [batch, output_len]
        return x
