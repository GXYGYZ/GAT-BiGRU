import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.nn import GATConv
import matplotlib as mpl
import matplotlib.font_manager as fm
import shutil
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
from tqdm import tqdm
import matplotlib as mpl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_in, n_out =10,5
batch_size = 1024
epochs = 100
data_folder = "数据保留2/all"
seed = 42


PHYSIO_CONNECTIONS = [
    (0, 7), (0, 10), (0, 6),
    (0, 4), (0, 8), (0, 11),
    (1, 7), (2, 9), (3, 4),
    (4, 10), (5, 6), (6, 10),
    (8, 9), (8, 10), (10, 11),
    (11, 12),
]

# 可用的静态特征：0-环境温度, 1-性别, 2-年龄, 3-身高, 4-体重, 5-BMI, 6-Clo
SELECTED_STATIC_FEATURES = [1,2,3,4,5]
def setup_chinese_font():

    try_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']


    for font in try_fonts + fm.findSystemFonts():
        if 'SimHei' in font or 'Microsoft YaHei' in font or 'msyh' in font.lower():

            plt.rcParams['font.sans-serif'] = [font.split('\\')[-1].split('.')[0]]
            plt.rcParams['axes.unicode_minus'] = False


            if os.path.exists(font):

                plt.rcParams['pdf.fonttype'] = 42  # 使用Type 3字体解决嵌入问题
                try:
                    font_prop = fm.FontProperties(fname=font)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    print(f"使用字体: {font}")
                    return
                except:
                    continue
            else:
                try:

                    plt.rcParams['font.sans-serif'] = [font.split('\\')[-1].split('.')[0]]
                    print(f"使用字体: {font}")
                    return
                except:
                    continue


    try:

        droid_font = 'DroidSansFallback.ttf'
        if not os.path.exists(droid_font):

            system_fonts = fm.findSystemFonts()
            for f in system_fonts:
                if 'droid' in f.lower() and 'sans' in f.lower():
                    droid_font = f
                    break
            else:

                import urllib.request
                print("正在下载开源中文字体...")
                urllib.request.urlretrieve(
                    "https://github.com/googlefonts/DroidSansFallback/raw/main/DroidSansFallback.ttf",
                    droid_font
                )


        font_prop = fm.FontProperties(fname=droid_font)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['pdf.fonttype'] = 42
        print(f"使用下载的字体: {droid_font}")
    except Exception as e:
        print(f"字体设置失败: {e}, 将使用默认英文")
        plt.rcParams['font.family'] = 'Arial'



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)



class SubjectDataset(Dataset):
    def __init__(self, subjects_data, dynamic_scaler, static_scaler, mode='train'):
        self.samples = []
        for subj in subjects_data:
            # 直接使用已处理好的静态特征
            static_processed = subj[mode]['static']
            dynamic = subj[mode]['dynamic']
            time_steps = subj[mode]['time']

            for i in range(len(dynamic) - n_in - n_out + 1):
                # 添加时间步长信息
                sample_time = time_steps[i:i + n_in]
                self.samples.append({
                    'X': dynamic[i:i + n_in],
                    'y': dynamic[i + n_in:i + n_in + n_out],
                    'static': static_processed,
                    'time': sample_time
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['X'], dtype=torch.float32),
            torch.tensor(sample['y'], dtype=torch.float32),
            torch.tensor(sample['time'], dtype=torch.long),
            torch.tensor(sample['static'], dtype=torch.float32)
        )
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timesteps):
        return self.pe[timesteps]

class TimeSeriesModel(nn.Module):
    def __init__(self,
                 dynamic_dim=13,
                 static_dim=7,
                 pos_enc_dim=32,
                 gat_layers=2,
                 gru_layers=2):
        super().__init__()
        self.static_dim = static_dim
        self.use_static = static_dim > 0  # 有静态信息才使用

        self.pos_encoder = SinusoidalPositionalEncoding(pos_enc_dim)

        # GAT参数
        self.gat_in = 1 + pos_enc_dim  # 原始特征+位置编码
        self.gat_hidden = 32
        self.gat_heads = 4


        self.gat = nn.ModuleList()
        self.gat_projections = nn.ModuleList()  # 残差投影
        for i in range(gat_layers):
            in_channels = self.gat_in if i == 0 else self.gat_hidden * self.gat_heads
            self.gat.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=self.gat_hidden,
                    heads=self.gat_heads,
                    concat=True,
                    dropout=0.2
                )
            )
            self.gat_projections.append(
                nn.Linear(in_channels, self.gat_hidden * self.gat_heads)
                if in_channels != self.gat_hidden * self.gat_heads
                else nn.Identity()
            )
            self.gat.append(nn.LayerNorm(self.gat_hidden * self.gat_heads))
            self.gat.append(nn.ReLU())


        self.raw_proj = nn.Sequential(
            nn.Linear(dynamic_dim, 128),  # 关键修改点1：输出维度改为128
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )


        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 256),  # 输入拼接后的总维度
            nn.Sigmoid()
        )

        if self.use_static:
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )
        else:
            self.static_net = None


        gru_input_size = 128 + 128  # raw_features + gat_seq
        if self.use_static:
            gru_input_size += 64  # 如果有静态特征再加64维

        self.bigru = nn.GRU(
            input_size=gru_input_size,  # 根据情况调整
            hidden_size=256,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if gru_layers > 1 else 0
        )
        self.gru_norm = nn.LayerNorm(256 * 2)


        self.output_net = nn.Sequential(
            ResidualBlock(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, dynamic_dim * n_out)
        )


        self.edge_index = self.create_edge_index().to(torch.long)
        self._init_weights()

    def create_edge_index(self):
        edges = []
        for u, v in PHYSIO_CONNECTIONS:
            edges.extend([[u, v], [v, u]])
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gat' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
                elif 'fc' in name or 'proj' in name:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, time_steps, static):
        batch_size, seq_len, _ = x.size()


        raw_features = self.raw_proj(x.reshape(-1, x.size(-1)))
        raw_features = raw_features.view(batch_size, seq_len, -1)  # [B, T, 128]


        pos_emb = self.pos_encoder(time_steps)  # [batch, seq, pos_dim]


        if self.use_static:
            static_feat = self.static_net(static)  # [batch, 64]
            static_expanded = static_feat.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, T, 64]
        else:
            static_expanded = None  # 无静态信息


        node_features = torch.cat([
            x.unsqueeze(-1),  # [batch, seq, 13, 1]
            pos_emb.unsqueeze(2).expand(-1, -1, 13, -1)  # [batch, seq, 13, pos_dim]
        ], dim=-1)  # [batch, seq, 13, 1+pos_dim]


        gat_outputs = []
        for t in range(seq_len):
            current_feat = node_features[:, t, :, :]  # [batch, 13, 1+pos_dim]
            num_nodes = 13


            offsets = torch.arange(batch_size, device=device) * num_nodes
            batch_edge_index = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
            batch_edge_index = batch_edge_index.view(2, -1)


            gat_input = current_feat.reshape(-1, self.gat_in)
            for i in range(0, len(self.gat), 3):
                conv, norm, act = self.gat[i], self.gat[i + 1], self.gat[i + 2]
                proj = self.gat_projections[i // 3]


                residual = proj(gat_input)
                gat_output = conv(gat_input, batch_edge_index)
                gat_output = norm(gat_output)
                gat_output += residual
                gat_input = act(gat_output)


            gat_output = gat_input.view(batch_size, num_nodes, -1)  # [batch, 13, 128]
            gat_output = torch.mean(gat_output, dim=1)  # [batch, 128]
            gat_outputs.append(gat_output)


        gat_seq = torch.stack(gat_outputs, dim=1)  # [batch, seq, 128]
        fused_feature = torch.cat([raw_features, gat_seq], dim=-1)  # [B, T, 256]
        gate = self.fusion_gate(fused_feature)  # [B, T, 256]


        fused = gate * fused_feature  # [B, T, 256]


        if static_expanded is not None:
            fused_with_static = torch.cat([fused, static_expanded], dim=-1)  # [B, T, 320]
        else:
            fused_with_static = fused


        gru_out, _ = self.bigru(fused_with_static)  # [batch, seq, 512]
        gru_feat = self.gru_norm(gru_out[:, -1, :])  # [batch, 512]


        output = self.output_net(gru_feat)  # [batch, 13 * 5]
        return output.view(-1, n_out, 13)  # [batch, 5, 13]

class TimeSeriesModel1(nn.Module):
    def __init__(self,
                 dynamic_dim=13,
                 static_dim=7,
                 pos_enc_dim=32,
                 gat_layers=2,
                 gru_layers=2):
        super().__init__()
        self.static_dim = static_dim
        self.use_static = static_dim > 0

        self.pos_encoder = SinusoidalPositionalEncoding(pos_enc_dim)


        self.gat_in = 1 + pos_enc_dim
        self.gat_hidden = 32
        self.gat_heads = 4


        self.gat = nn.ModuleList()
        self.gat_projections = nn.ModuleList()
        for i in range(gat_layers):
            in_channels = self.gat_in if i == 0 else self.gat_hidden * self.gat_heads
            self.gat.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=self.gat_hidden,
                    heads=self.gat_heads,
                    concat=True,
                    dropout=0.2
                )
            )
            self.gat_projections.append(
                nn.Linear(in_channels, self.gat_hidden * self.gat_heads)
                if in_channels != self.gat_hidden * self.gat_heads
                else nn.Identity()
            )
            self.gat.append(nn.LayerNorm(self.gat_hidden * self.gat_heads))
            self.gat.append(nn.ReLU())


        self.raw_proj = nn.Sequential(
            nn.Linear(dynamic_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )


        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )


        if self.use_static:
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )
        else:
            self.static_net = None


        self.bigru = nn.GRU(
                input_size=256,
                hidden_size=256,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if gru_layers > 1 else 0
            )
        self.gru_norm = nn.LayerNorm(256 * 2)


        output_net_size=512
        if self.use_static:
            output_net_size += 64  # 如果有静态特征再加64维

        self.output_net = nn.Sequential(
            ResidualBlock(output_net_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, dynamic_dim * n_out)
        )

        self.edge_index = self.create_edge_index().to(torch.long)
        self._init_weights()

    def create_edge_index(self):
        edges = []
        for u, v in PHYSIO_CONNECTIONS:
            edges.extend([[u, v], [v, u]])
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gat' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
                elif 'fc' in name or 'proj' in name:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, time_steps, static):
        batch_size, seq_len, _ = x.size()


        raw_features = self.raw_proj(x.reshape(-1, x.size(-1)))
        raw_features = raw_features.view(batch_size, seq_len, -1)  # [B, T, 128]


        pos_emb = self.pos_encoder(time_steps)  # [batch, seq, pos_dim]




        node_features = torch.cat([
            x.unsqueeze(-1),  # [batch, seq, 13, 1]
            pos_emb.unsqueeze(2).expand(-1, -1, 13, -1)  # [batch, seq, 13, pos_dim]
        ], dim=-1)  # [batch, seq, 13, 1+pos_dim]


        gat_outputs = []
        for t in range(seq_len):
            current_feat = node_features[:, t, :, :]  # [batch, 13, 1+pos_dim]
            num_nodes = 13


            offsets = torch.arange(batch_size, device=device) * num_nodes
            batch_edge_index = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
            batch_edge_index = batch_edge_index.view(2, -1)


            gat_input = current_feat.reshape(-1, self.gat_in)
            for i in range(0, len(self.gat), 3):
                conv, norm, act = self.gat[i], self.gat[i + 1], self.gat[i + 2]
                proj = self.gat_projections[i // 3]


                residual = proj(gat_input)
                gat_output = conv(gat_input, batch_edge_index)
                gat_output = norm(gat_output)
                gat_output += residual
                gat_input = act(gat_output)


            gat_output = gat_input.view(batch_size, num_nodes, -1)  # [batch, 13, 128]
            gat_output = torch.mean(gat_output, dim=1)  # [batch, 128]
            gat_outputs.append(gat_output)


        gat_seq = torch.stack(gat_outputs, dim=1)  # [batch, seq, 128]
        fused_feature = torch.cat([raw_features, gat_seq], dim=-1)  # [B, T, 256]
        gate = self.fusion_gate(fused_feature)  # [B, T, 256]


        fused = gate * fused_feature  # [B, T, 256]


        # fused_with_static = torch.cat([fused, static_expanded], dim=-1)  # [B, T, 320]



        gru_out, _ = self.bigru(fused)  # [batch, seq_len, 512]
        gru_feat = self.gru_norm(gru_out[:, -1, :])  # [batch, 512] (last timestep)

        if self.use_static:
            static_feat = self.static_net(static)  # [batch, 64]
            fused_with_static = torch.cat([gru_feat, static_feat], dim=-1)  # [batch, 576]
        else:
            fused_with_static = gru_feat  # [batch, 512]

        output = self.output_net(fused_with_static)  # [batch, 13 * n_out]
        return output.view(-1, n_out, 13)  # [batch, n_out, 13]



class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        out = self.fc(x)
        return out + residual



def load_data():
    all_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.xlsx')])
    all_static = []

    for f in all_files:
        df = pd.read_excel(os.path.join(data_folder, f))


        static_features = [
            df.iloc[1, 0],  # 环境温度
            df['Gender'].values[0],  # 性别
            df['Age'].values[0],  # 年龄
            df['Height'].values[0],  # 身高
            df['Weight'].values[0],  # 体重
            df['BMI'].values[0],  # BMI
            df['clo'].values[0]  # Clo
        ]


        selected_static = [static_features[i] for i in SELECTED_STATIC_FEATURES]
        all_static.append(selected_static)


    if SELECTED_STATIC_FEATURES:
        static_scaler = StandardScaler().fit(all_static)
    else:
        static_scaler = None  # 没有静态特征时不使用标准化器


    train_dynamic = []
    for f in all_files[:int(len(all_files) * 0.8)]:
        df = pd.read_excel(os.path.join(data_folder, f))
        train_dynamic.append(df.iloc[:, 2:15].values)
    dynamic_scaler = StandardScaler().fit(np.vstack(train_dynamic))

    subjects = []
    for f in all_files:
        df = pd.read_excel(os.path.join(data_folder, f))

        static_raw = [
            df.iloc[0, 0],
            df['Gender'].values[0],
            df['Age'].values[0],
            df['Height'].values[0],
            df['Weight'].values[0],
            df['BMI'].values[0],
            df['clo'].values[0]
        ]

        static_selected = [static_raw[i] for i in SELECTED_STATIC_FEATURES]


        if static_scaler is not None and static_selected:
            static_processed = static_scaler.transform([static_selected])[0]
        else:
            static_processed = np.array([], dtype=np.float32)

        dynamic = dynamic_scaler.transform(df.iloc[:, 2:15].values)
        n_total = len(dynamic)
        train_end = int(n_total * 0.5)
        val_end = int(n_total * 0.7)

        subjects.append({
            'train': {'dynamic': dynamic[:train_end], 'time': np.arange(train_end), 'static': static_processed},
            'val': {'dynamic': dynamic[train_end:val_end], 'time': np.arange(train_end, val_end),
                    'static': static_processed},
            'test': {'dynamic': dynamic[val_end:], 'time': np.arange(val_end, n_total), 'static': static_processed}
        })


    return subjects, dynamic_scaler, static_scaler, len(SELECTED_STATIC_FEATURES)



def evaluate_model(model, loader, dynamic_scaler):
    model.eval()
    all_preds = []
    all_trues = []
    output_preds = []
    output_trues = []

    with torch.no_grad():
        for X, y, t,s in loader:
            X, y, t, s = X.to(device), y.to(device), t.to(device), s.to(device)
            pred = model(X, t, s)

            # 逆标准化并保持三维结构
            X_real = dynamic_scaler.inverse_transform(X.cpu().numpy().reshape(-1, 13)).reshape(X.shape)
            y_real = dynamic_scaler.inverse_transform(y.cpu().numpy().reshape(-1, 13)).reshape(y.shape)
            pred_real = dynamic_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 13)).reshape(pred.shape)

            # 拼接完整序列
            for i in range(X.shape[0]):
                full_true = np.concatenate([X_real[i], y_real[i]])
                full_pred = np.concatenate([X_real[i], pred_real[i]])
                all_trues.append(full_true)
                all_preds.append(full_pred)
                output_trues.append(y_real[i])
                output_preds.append(pred_real[i])


    output_trues = np.stack(output_trues)
    output_preds = np.stack(output_preds)


    def calc_feature_wise(func):
        return [func(output_trues[..., i], output_preds[..., i]) for i in range(13)]

    metrics = {
        'MSE': lambda t, p: np.mean((t - p) ** 2),
        'MAE': lambda t, p: np.mean(np.abs(t - p)),
        'MAPE': lambda t, p: 100 * np.mean(np.abs((t - p) / (np.abs(t) + 1e-6))),
        'R2': lambda t, p: 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
    }

    results = {}
    for name, func in metrics.items():
        results[name] = np.array(calc_feature_wise(func))

    return results, np.array(all_preds), np.array(all_trues)



def visualize_metrics(results, feature_names):
    metrics = ['MSE', 'MAE', 'MAPE', 'R2']
    units = ['', '', '%', '']

    plt.figure(figsize=(18, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(feature_names)))

    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        values = results[metric]

        bars = plt.bar(range(len(values)), values, color=colors)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.title(f"{metric} ({units[idx - 1]})" if units[idx - 1] else metric)

        for bar, v in zip(bars, values):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval,
                     f'{v:.4f}{"%" if metric == "MAPE" else ""}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42  # 确保PDF嵌入字体


os.makedirs("results/pdp_all_nodes", exist_ok=True)
os.makedirs("results/pdp_per_node", exist_ok=True)



NODE_NAMES = [
    'Chest', 'Forehead', 'Instep', 'LeftBackLowLeg', 'LeftBackThigh',
    'LeftHand', 'LowArm', 'Neck', 'RightFrontThigh', 'RightLowLeg',
    'Scapula', 'UpperArm', 'Wrist'
]


def main():
    set_seed(seed)
    setup_chinese_font()

    subjects, dynamic_scaler, static_scaler, static_dim = load_data()

    train_set = SubjectDataset(subjects, dynamic_scaler, static_scaler, 'train')
    val_set = SubjectDataset(subjects, dynamic_scaler, static_scaler, 'val')
    test_set = SubjectDataset(subjects, dynamic_scaler, static_scaler, 'test')


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             worker_init_fn=worker_init_fn)


    model = TimeSeriesModel(static_dim=static_dim).to(device)  # 使用实际的静态特征维度
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()


    best_val_loss = float('inf')

    loss_history = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Val Loss'])

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for X, y, t, s in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            X, y, t, s = X.to(device), y.to(device), t.to(device), s.to(device)
            optimizer.zero_grad()
            output = model(X, t, s)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()


        avg_train_loss = train_loss / len(train_loader)


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y, t, s in val_loader:
                X, y, t, s = X.to(device), y.to(device), t.to(device), s.to(device)
                output = model(X, t, s)
                val_loss += criterion(output, y).item()


        avg_val_loss = val_loss / len(val_loader)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')


        loss_history = loss_history.append({
            'Epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Val Loss': avg_val_loss
        }, ignore_index=True)
        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    loss_history.to_excel('traini'
                          'ng_loss_history.xlsx', index=False)


    model.load_state_dict(torch.load('best_model.pth'))
    feature_names = ['Chest', 'Forehead', 'Instep', 'LeftBackLowLeg', 'LeftBackThigh',
                     'LeftHand', 'LowArm', 'Neck', 'RightFrontThigh', 'RightLowLeg',
                     'Scapula', 'UpperArm', 'Wrist']

    test_results, preds, trues = evaluate_model(model, test_loader, dynamic_scaler)

    print("\nTest Metrics Summary:")
    for metric in ['MSE', 'MAE', 'MAPE', 'R2']:
        print(f"{metric}: {test_results[metric].mean():.2f} ± {test_results[metric].std():.2f}")

    visualize_metrics(test_results, feature_names)


    result_df = pd.DataFrame({
        'Feature': feature_names,
        **{k: test_results[k] for k in ['MSE', 'MAE', 'MAPE', 'R2']}
    })
    result_df.to_csv('test_metrics.csv', index=False)
    print("\n详细指标已保存至 test_metrics.csv")

    model.load_state_dict(torch.load('best_model.pth'))

if __name__ == "__main__":
    main()

