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
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_in, n_out = 10, 5
batch_size = 4028
epochs = 200
data_folder = "数据保留3/all"

seed = 42


train_mask_mode = 'fixed'  # 'fixed'或'random'
fixed_mask_cols = [4]  
n_warmup = 60
n_medium = 130


QUANTILES = [0.05, 0.5, 0.95] 
N_QUANTILES = len(QUANTILES)
ALPHA = 0.1  
INTERVAL_WEIGHT = 0.4
CENTER_WEIGHT = 0.1
SPATIAL_WEIGHT = 0.1


DYNAMIC_COLS = ['Chest_Tsk', 'Forehead_Tsk', 'Instep_Tsk', 'LeftBackLowLeg_Tsk', 'LeftBackThigh_Tsk',
                'LeftHand_Tsk', 'LowArm_Tsk', 'Neck_Tsk', 'RightFrontThigh_Tsk', 'RightLowLeg_Tsk',
                'Scapula_Tsk', 'UpperArm_Tsk', 'Wrist_Tsk']

STATIC_CONT_COLS = ['Age', 'Height', 'Weight', 'BMI']  
STATIC_CAT_COLS = ['Gender']  



PHYSIO_CONNECTIONS = [
    (0, 7), (0, 10), (0, 6), (0, 4), (0, 8), (0, 11),
    (1, 7), (2, 9), (3, 4), (4, 10), (5, 6), (6, 10),
    (8, 9), (8, 10), (10, 11), (11, 12), (7, 10)
]



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class SubjectDataset(Dataset):
    def __init__(self, subjects_data, dynamic_scaler, static_scaler, mode='train'):
        self.samples = []
        for subj in subjects_data:
        
            static_processed = subj[mode]['static']
            dynamic = subj[mode]['dynamic']
            time_steps = subj[mode]['time']
            for i in range(len(dynamic) - n_in - n_out + 1):
                self.samples.append({
                    'X': dynamic[i:i + n_in],
                    'y': dynamic[i + n_in:i + n_in + n_out],
                    'time': time_steps[i:i + n_in],
                    'static': static_processed
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
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                             (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timesteps):
        if timesteps.dim() == 1:
            return self.pe[timesteps]
        else:
            return self.pe[timesteps.view(-1)].view(*timesteps.shape, -1)


def apply_train_mask(X, epoch, mode='random'):
   
    masked_X = X.clone()

    if mode == 'fixed':
        masked_X[:, :, fixed_mask_cols] = 0
    elif mode == 'random':
        if epoch < n_warmup:
            random_rate = 0.1
        elif epoch < n_warmup + n_medium:
            random_rate = 0.2
        else:
            random_rate = 0.3

        if random_rate > 0:
            batch, seq, feat = X.shape
            for b in range(batch):
                for t in range(seq):
                    n_mask = int(feat * random_rate)
                    if n_mask > 0:
                        cols = np.random.choice(feat, size=n_mask, replace=False)
                        masked_X[b, t, cols] = 0
    return masked_X


def apply_test_mask(X, mask_cols=None, random_rate=0.0):
  
    masked_X = X.clone()

    if mask_cols:
        masked_X[:, :, mask_cols] = 0
    if random_rate > 0:
        batch, seq, feat = X.shape
        for b in range(batch):
            for t_idx in range(seq):
                n_mask = int(feat * random_rate)
                if n_mask > 0:
                    cols = np.random.choice(feat, size=n_mask, replace=False)
                    masked_X[b, t_idx, cols] = 0
    return masked_X


class GATRecoveryNet(nn.Module):
    def __init__(self, dynamic_dim=13, pos_enc_dim=32, output_dim=128):
        super().__init__()
        self.pos_encoder = SinusoidalPositionalEncoding(pos_enc_dim)
        self.gat1 = GATConv(in_channels=pos_enc_dim + 1, out_channels=64, heads=4)
        self.gat2 = GATConv(in_channels=64 * 4, out_channels=64, heads=2)

       
        self.spatial_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  
        )

     
        self.feature_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128)
        )

        self.edge_index = self.create_batch_edge_index(batch_size=1)

    def create_batch_edge_index(self, batch_size):
        base_edges = []
        for u, v in PHYSIO_CONNECTIONS:
            base_edges.extend([(u, v), (v, u)])
        edge_indices = []
        for b in range(batch_size):
            offset = b * 13
            for u, v in base_edges:
                edge_indices.append([u + offset, v + offset])
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)

    def forward(self, x, timesteps):
        B, T, _ = x.size()
        pos_emb = self.pos_encoder(timesteps)

       
        node_feat = torch.cat([
            x.unsqueeze(-1),  # [B, T, 13, 1]
            pos_emb.unsqueeze(2).expand(-1, -1, 13, -1)  # [B, T, 13, pos_enc_dim]
        ], dim=-1)
        node_feat = node_feat.view(-1, node_feat.size(-1))

        
        self.edge_index = self.create_batch_edge_index(batch_size=B * T)

        node_feat = F.relu(self.gat1(node_feat, self.edge_index))
        node_feat = F.relu(self.gat2(node_feat, self.edge_index))
        node_feat = node_feat.view(B, T, 13, -1)  # [B, T, 13, 128]

     
        temporal_features = self.feature_net(node_feat)  # [B, T, 13, 128]

   
        base_temps = self.spatial_net(node_feat)  # [B, T, 13, 1]
        base_temps = base_temps.squeeze(-1)  

        return base_temps, temporal_features.mean(dim=2) 


class EnhancedTimeSeriesModelV3(nn.Module):
    def __init__(self, dynamic_dim=13, static_dim=5, pos_enc_dim=32):
        super().__init__()
        self.pos_encoder = SinusoidalPositionalEncoding(pos_enc_dim)
        self.spatial_net = GATRecoveryNet(dynamic_dim, pos_enc_dim)

        
        self.masked_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

      
        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )

       
        self.temporal_net = nn.GRU(
            input_size=256, hidden_size=256,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.3
        )

       
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        
        self.delta_output_net = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, dynamic_dim * n_out * N_QUANTILES)  
        )

    def forward(self, X_masked, time_steps, static, return_base=False):
        batch_size, seq_len, _ = X_masked.size()

       
        base_temps, spatial_features = self.spatial_net(X_masked, time_steps) 

   
        masked_encoded = self.masked_encoder(X_masked.view(-1, 13))
        masked_encoded = masked_encoded.view(batch_size, seq_len, -1)

        
        fused_feat = torch.cat([masked_encoded, spatial_features], dim=-1)
        gate = self.fusion_gate(fused_feat)
        fused = gate * fused_feat

       
        gru_out, _ = self.temporal_net(fused)
        gru_feat = gru_out[:, -1, :] 

        
        static_feat = self.static_net(static)
        combined = torch.cat([gru_feat, static_feat], dim=-1)

        
        delta = self.delta_output_net(combined)  # [B, 13*n_out*3]
        delta = delta.view(batch_size, n_out, 13, N_QUANTILES)  # [B, n_out, 13, 3]

        
        base = base_temps[:, -1:, :]  # [B, 1, 13]
        base = base.unsqueeze(-1).expand(-1, n_out, 13, N_QUANTILES)  # [B, n_out, 13, 3]

      
        final_pred = base + delta  # [B, n_out, 13, 3]

        if return_base:
            return final_pred, base, delta
        return final_pred


def compute_quantile_loss(pred_quantiles, target):
  
    quantile_loss = 0
    for i, q in enumerate(QUANTILES):
        error = target - pred_quantiles[..., i]
        loss = torch.max(q * error, (q - 1) * error)
        quantile_loss += loss.mean()
    return quantile_loss


def compute_interval_score(pred_quantiles, target, alpha=ALPHA):
   
    lower = pred_quantiles[..., 0]  
    upper = pred_quantiles[..., 2]  

    interval_width = upper - lower
    below_lower = torch.clamp(lower - target, min=0)
    above_upper = torch.clamp(target - upper, min=0)

    interval_score = interval_width.mean() + (2 / alpha) * (below_lower + above_upper).mean()
    return interval_score


def compute_center_loss(pred_quantiles, target):
    
    lower = pred_quantiles[..., 0]  
    upper = pred_quantiles[..., 2]  
    median = pred_quantiles[..., 1]  

    
    interval_center = (lower + upper) / 2

    
    center_loss = F.mse_loss(median, interval_center)

    return center_loss


def compute_spatial_loss(pred_quantiles, X_masked_last):
   
    median = pred_quantiles[..., 1] 
    current_temp_masked = X_masked_last[:, -1:, :]  
    current_temp_expanded = current_temp_masked.expand(-1, n_out, -1)

    spatial_loss = F.mse_loss(median, current_temp_expanded)
    return spatial_loss


def triple_loss_function(pred_quantiles, target, X_masked_last):
    
    quantile_loss = compute_quantile_loss(pred_quantiles, target)

    
    interval_score = compute_interval_score(pred_quantiles, target, alpha=ALPHA)

    center_loss = compute_center_loss(pred_quantiles, target)

   
    spatial_loss = compute_spatial_loss(pred_quantiles, X_masked_last)

 
    lower = pred_quantiles[..., 0]  
    upper = pred_quantiles[..., 2]  
    median = pred_quantiles[..., 1] 

   
    coverage = ((target >= lower) & (target <= upper)).float().mean()
    interval_width = (upper - lower).mean()

   
    total_loss = (
            quantile_loss +
            INTERVAL_WEIGHT * interval_score +  
            CENTER_WEIGHT * center_loss +  
            SPATIAL_WEIGHT * spatial_loss  
    )

    loss_info = {
        'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'quantile_loss': quantile_loss.item(),
        'interval_score': interval_score.item(),
        'center_loss': center_loss.item(),
        'spatial_loss': spatial_loss.item(),
        'coverage': coverage.item(),
        'interval_width': interval_width.item()
    }

    return total_loss, loss_info



def load_data():
    all_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.xlsx')])

   
    all_static_cont = []
    for f in all_files:
        df = pd.read_excel(os.path.join(data_folder, f))
        static_cont = df.iloc[0][STATIC_CONT_COLS].values
        all_static_cont.append(static_cont)
    static_scaler = StandardScaler().fit(np.vstack(all_static_cont))


    train_dynamic = []
    for f in all_files[:int(len(all_files) * 0.8)]:
        df = pd.read_excel(os.path.join(data_folder, f))
        train_dynamic.append(df[DYNAMIC_COLS].values)
    dynamic_scaler = StandardScaler().fit(np.vstack(train_dynamic))

    subjects = []
    for f in all_files:
        df = pd.read_excel(os.path.join(data_folder, f))

     
        static_cat = df.iloc[0][STATIC_CAT_COLS].values.astype(np.float32)
        static_cont = df.iloc[0][STATIC_CONT_COLS].values.astype(np.float32)
        static_cont_scaled = static_scaler.transform(static_cont.reshape(1, -1)).flatten()
        static_processed = np.concatenate([static_cat, static_cont_scaled])

      
        dynamic = dynamic_scaler.transform(df[DYNAMIC_COLS].values)

        n_total = len(dynamic)
        split_points = [int(n_total * 0.5), int(n_total * 0.7)]

        subjects.append({
            'train': {'dynamic': dynamic[:split_points[0]], 'time': np.arange(split_points[0]),
                      'static': static_processed},
            'val': {'dynamic': dynamic[split_points[0]:split_points[1]],
                    'time': np.arange(split_points[0], split_points[1]), 'static': static_processed},
            'test': {'dynamic': dynamic[split_points[1]:], 'time': np.arange(split_points[1], n_total),
                     'static': static_processed}
        })
    return subjects, dynamic_scaler, static_scaler



def evaluate(model, loader, dynamic_scaler, feature_names, mask_cols=None, random_rate=0.0):
    model.eval()

   
    all_preds = [] 
    all_trues = []  
    output_preds = []  
    output_trues = []  
    all_intervals = []  
    all_medians = []  

    metrics = {name: {'MSE': [], 'MAE': [], 'MAPE': [], 'R2': [],
                      'Coverage_90': [], 'Interval_Width': []} for name in feature_names}

    with torch.no_grad():
        for X_orig, y, t, s in loader:
            X_orig, y = X_orig.to(device), y.to(device)
            t, s = t.to(device), s.to(device)

          
            X_masked = apply_test_mask(X_orig, mask_cols, random_rate)

            
            pred_quantiles = model(X_masked, t, s)  # [B, n_out, 13, 3]

            
            pred_median = pred_quantiles[..., 1] 

            
            pred_lower = pred_quantiles[..., 0]  
            pred_upper = pred_quantiles[..., 2]  

            
            X_real = dynamic_scaler.inverse_transform(X_orig.cpu().numpy().reshape(-1, 13)).reshape(X_orig.shape)
            y_real = dynamic_scaler.inverse_transform(y.cpu().numpy().reshape(-1, 13)).reshape(y.shape)
            pred_median_real = dynamic_scaler.inverse_transform(
                pred_median.detach().cpu().numpy().reshape(-1, 13)).reshape(pred_median.shape)
            pred_lower_real = dynamic_scaler.inverse_transform(
                pred_lower.detach().cpu().numpy().reshape(-1, 13)).reshape(pred_lower.shape)
            pred_upper_real = dynamic_scaler.inverse_transform(
                pred_upper.detach().cpu().numpy().reshape(-1, 13)).reshape(pred_upper.shape)

            
            for i in range(X_orig.shape[0]):
                full_true = np.concatenate([X_real[i], y_real[i]])
                full_pred = np.concatenate([X_real[i], pred_median_real[i]])
                all_trues.append(full_true)
                all_preds.append(full_pred)
                output_trues.append(y_real[i])
                output_preds.append(pred_median_real[i])

             
                interval_info = {
                    'lower': pred_lower_real[i],
                    'upper': pred_upper_real[i],
                    'median': pred_median_real[i]
                }
                all_intervals.append(interval_info)
                all_medians.append(pred_median_real[i])

    
    output_trues = np.stack(output_trues)
    output_preds = np.stack(output_preds)

   
    for f_idx, name in enumerate(feature_names):
        true = output_trues[..., f_idx].flatten()
        pred = output_preds[..., f_idx].flatten()

       
        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))
        mape = 100 * np.mean(np.abs((true - pred) / (np.abs(true) + 1e-6)))
        r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)

       
        lower_all = np.stack([interval['lower'][:, f_idx] for interval in all_intervals]).flatten()
        upper_all = np.stack([interval['upper'][:, f_idx] for interval in all_intervals]).flatten()

        coverage = np.mean((true >= lower_all) & (true <= upper_all))
        interval_width = np.mean(upper_all - lower_all)

        metrics[name]['MSE'].append(mse)
        metrics[name]['MAE'].append(mae)
        metrics[name]['MAPE'].append(mape)
        metrics[name]['R2'].append(r2)
        metrics[name]['Coverage_90'].append(coverage)
        metrics[name]['Interval_Width'].append(interval_width)

    
    avg_metrics = {
        'MSE': np.mean([metrics[n]['MSE'][0] for n in feature_names]),
        'MAE': np.mean([metrics[n]['MAE'][0] for n in feature_names]),
        'MAPE': np.mean([metrics[n]['MAPE'][0] for n in feature_names]),
        'R2': np.mean([metrics[n]['R2'][0] for n in feature_names]),
        'Coverage_90': np.mean([metrics[n]['Coverage_90'][0] for n in feature_names]),
        'Interval_Width': np.mean([metrics[n]['Interval_Width'][0] for n in feature_names])
    }

    return metrics, avg_metrics, np.array(all_preds), np.array(all_trues), all_intervals, all_medians



def visualize_with_intervals(all_preds, all_trues, all_intervals, feature_names, title, num_samples=3):
    np.random.seed(seed)
    sample_indices = np.random.choice(len(all_preds), num_samples, replace=False)

    for idx, sample_idx in enumerate(sample_indices, 1):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        full_true = all_trues[sample_idx]  
        full_pred = all_preds[sample_idx]  

        
        interval_info = all_intervals[sample_idx]
        lower = interval_info['lower']  # [n_out, 13]
        upper = interval_info['upper']  # [n_out, 13]
        median = interval_info['median']  # [n_out, 13]

       
        selected_features = [0, 3, 6, 9, 12] 
        colors = plt.cm.Set3(np.linspace(0, 1, len(selected_features)))

        for ax_row in axes:
            for ax in ax_row:
                ax.axvline(x=n_in - 0.5, color='gray', linestyle='-', alpha=0.7, linewidth=1)

       
        for i, feat_idx in enumerate(selected_features):
            color = colors[i]
            feature_name = feature_names[feat_idx]

           
            axes[0, 0].plot(range(n_in), full_true[:n_in, feat_idx],
                            color=color, alpha=0.7, linewidth=2, label=f'{feature_name}')

            
            axes[0, 0].plot(range(n_in, n_in + n_out), median[:, feat_idx],
                            color=color, linewidth=2, linestyle='-')
            axes[0, 0].plot(range(n_in, n_in + n_out), full_true[n_in:, feat_idx],
                            color=color, linewidth=2, linestyle='--')

          
            axes[0, 0].fill_between(range(n_in, n_in + n_out),
                                    lower[:, feat_idx], upper[:, feat_idx],
                                    color=color, alpha=0.2)

        axes[0, 0].set_title('点预测 vs 真实值 (带90%预测区间)')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('温度值')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        
        for i, feat_idx in enumerate(selected_features):
            color = colors[i]
            feature_name = feature_names[feat_idx]

           
            true_vals = full_true[n_in:, feat_idx]
            in_interval = (true_vals >= lower[:, feat_idx]) & (true_vals <= upper[:, feat_idx])

            axes[0, 1].scatter(range(n_in, n_in + n_out), true_vals,
                               c=['green' if in_ else 'red' for in_ in in_interval],
                               label=f'{feature_name}', alpha=0.7)

            
            axes[0, 1].fill_between(range(n_in, n_in + n_out),
                                    lower[:, feat_idx], upper[:, feat_idx],
                                    color=color, alpha=0.1)
            axes[0, 1].plot(range(n_in, n_in + n_out), median[:, feat_idx],
                            color=color, linewidth=1, alpha=0.5)

        axes[0, 1].set_title('预测区间覆盖情况 (绿色: 在区间内, 红色: 在区间外)')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('温度值')
        axes[0, 1].grid(True, alpha=0.3)

      
        for i, feat_idx in enumerate(selected_features):
            color = colors[i]
            feature_name = feature_names[feat_idx]

            interval_width = upper[:, feat_idx] - lower[:, feat_idx]
            axes[1, 0].plot(range(n_in, n_in + n_out), interval_width,
                            color=color, marker='o', markersize=4, label=feature_name)

        axes[1, 0].set_title('预测区间宽度')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('区间宽度 (℃)')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

       
        errors = []
        for i, feat_idx in enumerate(selected_features):
            pred_vals = median[:, feat_idx]
            true_vals = full_true[n_in:, feat_idx]
            error = pred_vals - true_vals
            errors.extend(error)

            axes[1, 1].hist(error, bins=20, alpha=0.5,
                            label=f'{feature_names[feat_idx]}', color=colors[i])

        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('预测误差分布')
        axes[1, 1].set_xlabel('预测误差 (℃)')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{title} - 样本 {idx}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{title}_interval_sample_{idx}.png', dpi=300, bbox_inches='tight')
        plt.show()



def main():
    set_seed(seed)
    subjects, dynamic_scaler, static_scaler = load_data()

    
    feature_names = [col.replace('_Tsk', '') for col in DYNAMIC_COLS]

    
    train_set = SubjectDataset(subjects, dynamic_scaler, static_scaler, 'train')
    val_set = SubjectDataset(subjects, dynamic_scaler, static_scaler, 'val')
    test_set = SubjectDataset(subjects, dynamic_scaler, static_scaler, 'test')

   
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size)
    test_loader = DataLoader(test_set, batch_size)

   
    model = EnhancedTimeSeriesModelV3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    best_val_loss = float('inf')
    best_epoch = 0

   
    history = {
        'train_loss': [], 'val_loss': [],
        'quantile_loss': [], 'interval_score': [],
        'center_loss': [], 'spatial_loss': [],
        'coverage': [], 'interval_width': [],
        'learning_rate': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        quantile_loss_total = 0
        interval_score_total = 0
        center_loss_total = 0
        spatial_loss_total = 0
        coverage_total = 0
        interval_width_total = 0

        for X_orig, y, t, s in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_orig, y = X_orig.to(device), y.to(device)
            t, s = t.to(device), s.to(device)

           
            X_masked = apply_train_mask(X_orig, epoch, mode=train_mask_mode)

           
            pred_quantiles = model(X_masked, t, s)  # [B, n_out, 13, 3]

          
            loss, loss_info = triple_loss_function(pred_quantiles, y, X_masked)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss_info['total_loss']
            quantile_loss_total += loss_info['quantile_loss']
            interval_score_total += loss_info['interval_score']
            center_loss_total += loss_info['center_loss']
            spatial_loss_total += loss_info['spatial_loss']
            coverage_total += loss_info['coverage']
            interval_width_total += loss_info['interval_width']

       
        model.eval()
        val_loss = 0
        val_quantile_loss = 0
        val_interval_score = 0
        val_center_loss = 0
        val_spatial_loss = 0
        val_coverage = 0
        val_interval_width = 0

        with torch.no_grad():
            for X_orig, y, t, s in val_loader:
                X_orig, y = X_orig.to(device), y.to(device)
                t, s = t.to(device), s.to(device)

               
                X_masked_val = X_orig.clone()  

                pred_quantiles = model(X_masked_val, t, s)  # [B, n_out, 13, 3]
                loss, loss_info = triple_loss_function(pred_quantiles, y, X_masked_val)

                val_loss += loss_info['total_loss']
                val_quantile_loss += loss_info['quantile_loss']
                val_interval_score += loss_info['interval_score']
                val_center_loss += loss_info['center_loss']
                val_spatial_loss += loss_info['spatial_loss']
                val_coverage += loss_info['coverage']
                val_interval_width += loss_info['interval_width']

        
        avg_train_loss = train_loss / len(train_loader)
        avg_quantile_loss = quantile_loss_total / len(train_loader)
        avg_interval_score = interval_score_total / len(train_loader)
        avg_center_loss = center_loss_total / len(train_loader)
        avg_spatial_loss = spatial_loss_total / len(train_loader)
        avg_coverage = coverage_total / len(train_loader)
        avg_interval_width = interval_width_total / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_quantile_loss = val_quantile_loss / len(val_loader)
        avg_val_interval_score = val_interval_score / len(val_loader)
        avg_val_center_loss = val_center_loss / len(val_loader)
        avg_val_spatial_loss = val_spatial_loss / len(val_loader)
        avg_val_coverage = val_coverage / len(val_loader)
        avg_val_interval_width = val_interval_width / len(val_loader)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model_v3_quantile_simple.pth')

        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['quantile_loss'].append(avg_quantile_loss)
        history['interval_score'].append(avg_interval_score)
        history['center_loss'].append(avg_center_loss)
        history['spatial_loss'].append(avg_spatial_loss)
        history['coverage'].append(avg_coverage)
        history['interval_width'].append(avg_interval_width)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

       
        if train_mask_mode == 'fixed':
            mask_info = f"固定掩码列{fixed_mask_cols}"
        else:
            if epoch < n_warmup:
                mask_info = "无随机掩码"
            elif epoch < n_warmup + n_medium:
                mask_info = "20%随机掩码"
            else:
                mask_info = "50%随机掩码"

        print(f"\nEpoch {epoch + 1:03d}/{epochs} | 策略: {mask_info}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Quantile Loss: {avg_quantile_loss:.4f} | Interval Score: {avg_interval_score:.4f}")
        print(f"  Center Loss: {avg_center_loss:.4f} | Spatial Loss: {avg_spatial_loss:.4f}")
        print(f"  Coverage: {avg_coverage:.3f} | Interval Width: {avg_interval_width:.4f}")
        print(f"  Val Coverage: {avg_val_coverage:.3f} | Val Width: {avg_val_interval_width:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

 
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

   
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('总损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

   
    axes[0, 1].plot(history['quantile_loss'], label='分位数损失', color='green', linewidth=1.5)
    axes[0, 1].plot(history['interval_score'], label='区间评分', color='orange', linewidth=1.5)
    axes[0, 1].plot(history['center_loss'], label='中心对齐损失', color='purple', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('损失组件')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

 
    axes[1, 0].plot(history['spatial_loss'], label='空间一致性损失', color='brown', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('空间一致性损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    
    axes[1, 1].plot(history['coverage'], label='训练覆盖率', color='green', linewidth=2)
    axes[1, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='目标覆盖率(90%)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('覆盖率')
    axes[1, 1].set_title('预测区间覆盖率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    
    axes[2, 0].plot(history['interval_width'], label='区间宽度', color='orange', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('宽度')
    axes[2, 0].set_title('平均预测区间宽度')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    
    axes[2, 1].plot(history['learning_rate'], label='学习率', color='purple', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('学习率')
    axes[2, 1].set_title('学习率变化')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('training_history_v3_quantile_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")

   
    print("\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load('best_model_v3_quantile_simple.pth'))

    test_strategies = [
        # ("无掩码", None, 0.0),
        ("固定列掩码", fixed_mask_cols, 0.0),
        #  ("随机30%掩码", None, 0.5),
    ]

    for name, mask_cols, random_rate in test_strategies:
        print(f"\n{'=' * 50}")
        print(f"测试策略: {name}")
        print('=' * 50)

        metrics, avg_metrics, all_preds, all_trues, all_intervals, all_medians = evaluate(
            model, test_loader, dynamic_scaler, feature_names,
            mask_cols=mask_cols, random_rate=random_rate
        )

        print(f"平均MSE: {avg_metrics['MSE']:.4f} | 平均MAE: {avg_metrics['MAE']:.4f}℃")
        print(f"平均MAPE: {avg_metrics['MAPE']:.2f}% | 平均R²: {avg_metrics['R2']:.4f}")
        print(f"平均覆盖率: {avg_metrics['Coverage_90']:.3f} | 平均区间宽度: {avg_metrics['Interval_Width']:.4f}℃")

        
        result_df = pd.DataFrame([
            {
                'Feature': f_name,
                'MSE': metrics[f_name]['MSE'][0],
                'MAE': metrics[f_name]['MAE'][0],
                'MAPE': metrics[f_name]['MAPE'][0],
                'R2': metrics[f_name]['R2'][0],
                'Coverage_90': metrics[f_name]['Coverage_90'][0],
                'Interval_Width': metrics[f_name]['Interval_Width'][0]
            } for f_name in feature_names
        ])
        result_df.to_csv(f'{name}_quantile_metrics_simple.csv', index=False)

        avg_row = {
            'Feature': 'Average',
            'MSE': avg_metrics['MSE'],
            'MAE': avg_metrics['MAE'],
            'MAPE': avg_metrics['MAPE'],
            'R2': avg_metrics['R2'],
            'Coverage_90': avg_metrics['Coverage_90'],
            'Interval_Width': avg_metrics['Interval_Width']
        }

        avg_df = pd.DataFrame([avg_row])
        result_df = pd.concat([result_df, avg_df], ignore_index=True)
        result_df.to_csv(f'{name}_quantile_metrics_with_avg_simple.csv', index=False)

        
        visualize_with_intervals(all_preds, all_trues, all_intervals,
                                 feature_names, name, num_samples=2)

        print(f"{name}测试完成，结果已保存")


if __name__ == "__main__":
    main()
