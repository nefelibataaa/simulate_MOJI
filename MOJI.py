import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score, accuracy_score

# 读取 CSV 文件
data = pd.read_csv("E:/python_project/MOJI/jscode.csv")
# 提取数值表示和标签
X = data['NumericalRepresentation'].apply(eval).tolist()
y = data['Label'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 自定义数据集
class CodeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = CodeDataset(X_train, y_train)
test_dataset = CodeDataset(X_test, y_test)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# MOJI定义
class MOJIModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=1):
        super(MOJIModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.2)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=ks)
            for ks in [2, 4, 6, 8]
        ])
        
        self.fc1 = nn.Linear(4 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout_fc = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pool_outs = [F.adaptive_max_pool1d(out, 1).squeeze(2) for out in conv_outs]

        x = torch.cat(pool_outs, dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.layer_norm(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        return torch.sigmoid(x)

# 模型参数
vocab_size = 1114111
embed_dim = 64

# 创建模型
model = MOJIModel(vocab_size=vocab_size, embed_dim=embed_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练阶段
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze(1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')

# 测试阶段
model.eval()
test_loss = 0
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze(1)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
        preds = (outputs >= 0.5).float()
        y_true.extend(y_batch.tolist())
        y_pred.extend(preds.tolist())
avg_test_loss = test_loss / len(test_loader)

# 计算性能指标
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')