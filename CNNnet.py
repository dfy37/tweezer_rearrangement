import os
import json
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class AtomRearrangementNet(nn.Module):
    def __init__(self, M):
        super(AtomRearrangementNet, self).__init__()
        
        # 接下来写AlexNet CNN部分
        

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入通道1，输出通道32，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入通道32，输出通道64，3x3卷积核
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输入通道64，输出通道128，3x3卷积核
        
        # 计算CNN输出的flatten后的维度
        self.flatten_size = 128 * M * M  # 最后一个卷积层的输出通道数 * 输入矩阵的大小
        
        # 全连接层部分
        # 预测方向d（2个类别）
        self.fc_d = nn.Linear(self.flatten_size, 2)
        
        # 预测n_x（选择的行/列个数）和P_x（选择的行/列的概率分布）
        self.fc_nx = nn.Linear(self.flatten_size, M)  # n_x 是一个长度为M的one-hot编码
        self.fc_Px = nn.Linear(self.flatten_size, M)  # P_x 是一个长度为M的one-hot编码
        
        # 预测n_y（移动维度选择的个数）和P_y1/P_y2（移动维度的起始点和终点概率）
        self.fc_ny = nn.Linear(self.flatten_size, M)  # n_y 是一个长度为M的one-hot编码
        self.fc_Py1 = nn.Linear(self.flatten_size, M)  # P_y1 是一个长度为M的one-hot编码
        self.fc_Py2 = nn.Linear(self.flatten_size, M)  # P_y2 是一个长度为M的one-hot编码

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        # 输入x的维度是 (batch_size, 1, M, M)，即每个样本是一个M x M的矩阵
        x = F.relu(self.conv1(x))  # 卷积层1，ReLU激活
        x = F.relu(self.conv2(x))  # 卷积层2，ReLU激活
        x = F.relu(self.conv3(x))  # 卷积层3，ReLU激活
        
        # 将特征图展平（flatten）
        x = x.view(x.size(0), -1)  # (batch_size, 128 * M * M)
        
        # 预测方向d
        d = self.fc_d(x)  # 输出方向d的one-hot编码概率
        d = F.softmax(d, dim=1)  # 使用Softmax获得概率分布
        
        # 预测n_x（选择的行/列个数）和P_x（选择的行/列的概率分布）
        n_x = self.fc_nx(x)  # 输出选择的行/列个数（one-hot）
        P_x = self.fc_Px(x)  # 输出选择的行/列概率
        
        # 预测n_y（移动维度的选择个数）和P_y1/P_y2（移动维度的起始点和终点概率）
        n_y = self.fc_ny(x)  # 输出选择的移动维度个数（one-hot）
        P_y1 = self.fc_Py1(x)  # 输出移动维度起始点概率
        P_y2 = self.fc_Py2(x)  # 输出移动维度终点点概率
        
        # # 使用Softmax将所有概率转化为分布
        # n_x = F.softmax(n_x, dim=1)
        # P_x = F.softmax(P_x, dim=1)
        # n_y = F.softmax(n_y, dim=1)
        # P_y1 = F.softmax(P_y1, dim=1)
        # P_y2 = F.softmax(P_y2, dim=1)

        return d, n_x, P_x, n_y, P_y1, P_y2

class AtomRearrangementDataset(Dataset):
    def __init__(self, input_data, targets, dim=16):
        self.input_data = input_data  # 输入数据 (batch_size, 1, M, M)
        self.targets = targets        # 目标标签，包含d, n_x, P_x, n_y, P_y1, P_y2
        self.dim = dim
        for idx in range(len(self.targets)):
            # 希望将[1,3,4]变为[0, 1, 0, 1, 1, 0, ..., 0]的one-hot编码形式
            self.targets[idx]['P_x'] = F.one_hot(self.targets[idx]['P_x'].long(), num_classes=self.dim).squeeze(0).float()
            if len(self.targets[idx]['P_x'].shape) > 1:
                self.targets[idx]['P_x'] = self.targets[idx]['P_x'].sum(0)
            self.targets[idx]['P_y1'] = F.one_hot(self.targets[idx]['P_y1'].long(), num_classes=self.dim).squeeze(0).float()
            if len(self.targets[idx]['P_y1'].shape) > 1:
                self.targets[idx]['P_y1'] = self.targets[idx]['P_y1'].sum(0)
            self.targets[idx]['P_y2'] = F.one_hot(self.targets[idx]['P_y2'].long(), num_classes=self.dim).squeeze(0).float()
            if len(self.targets[idx]['P_y2'].shape) > 1:
                self.targets[idx]['P_y2'] = self.targets[idx]['P_y2'].sum(0)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.targets[idx]

# 训练过程
def train(model, train_loader, optimizer, epochs=10):
    model.train()  # 设置为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            # 清零梯度
            optimizer.zero_grad()

            # 将输入和标签移动到GPU（如果有的话）
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # 前向传播
            d, n_x, P_x, n_y, P_y1, P_y2 = model(inputs)
            # print(targets['P_x'])

            # 计算损失
            loss_d = criterion_d(d, targets['d'])
            loss_nx = criterion_nx(n_x, targets['n_x'])
            loss_ny = criterion_ny(n_y, targets['n_y'])
            loss_Px = criterion_Px(P_x, targets['P_x'])
            loss_Py1 = criterion_Py1(P_y1, targets['P_y1'])
            loss_Py2 = criterion_Py2(P_y2, targets['P_y2'])

            # 总损失
            total_loss = loss_d + loss_nx + loss_Px + loss_ny + loss_Py1 + loss_Py2

            # 反向传播
            total_loss.backward()

            # 优化
            optimizer.step()

            # 统计损失
            running_loss += total_loss.item()
            if i % 100 == 99:  # 每100个batch输出一次
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                print(loss_d.item(), loss_nx.item(), loss_Px.item(), loss_ny.item(), loss_Py1.item(), loss_Py2.item())
                print('P_y1', P_y1)
                print('target P_y1', targets['P_y1'])
                running_loss = 0.0

def validate(model, val_loader):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度，节省内存
        total_loss = 0.0
        for inputs, targets in val_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            d, n_x, P_x, n_y, P_y1, P_y2 = model(inputs)
            loss_d = criterion_d(d, targets['d'])
            loss_nx = criterion_nx(n_x, targets['n_x'])
            loss_ny = criterion_ny(n_y, targets['n_y'])
            loss_Px = criterion_Px(P_x, targets['P_x'])
            loss_Py1 = criterion_Py1(P_y1, targets['P_y1'])
            loss_Py2 = criterion_Py2(P_y2, targets['P_y2'])
            total_loss += (loss_d + loss_nx + loss_Px + loss_ny + loss_Py1 + loss_Py2).item()
        print(f"Validation Loss: {total_loss / len(val_loader):.4f}")

def read_data(root):
    inputs = []
    targets = []

    paths = glob.glob(os.path.join(root, '*.json'))
    for path in paths[:100]:
        with open(path, 'r') as f:
            file_data = json.load(f)
            for d in file_data:
                inputs.append(torch.tensor(d['state'], dtype=torch.float32))  # 添加通道维度
                targets.append({
                    'd': torch.tensor(d['operation']['axis'], dtype=torch.long),
                    'n_x': torch.tensor(len(d['operation']['fixed_indices']), dtype=torch.long),
                    'P_x': torch.tensor(d['operation']['fixed_indices'], dtype=torch.float32),
                    'n_y': torch.tensor(len(d['operation']['select1']), dtype=torch.long),
                    'P_y1': torch.tensor(d['operation']['select1'], dtype=torch.float32),
                    'P_y2': torch.tensor(d['operation']['select2'], dtype=torch.float32),
                })

    return inputs, targets

if __name__ == '__main__':
    M = 16  # 假设阵列的大小是 M x M
    model = AtomRearrangementNet(M)
    
    inputs, targets = read_data('/Users/duanfeiyu/Documents/AtomRL/PathJson')
    
    train_dataset = AtomRearrangementDataset(inputs, targets)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    criterion_d = nn.CrossEntropyLoss()  # 方向预测，二分类
    criterion_nx = nn.CrossEntropyLoss()  # 选择的行/列个数n_x，one-hot编码
    criterion_ny = nn.CrossEntropyLoss()  # 选择的移动维度个数n_y，one-hot编码
    criterion_Px = nn.MSELoss()  # 选择的行/列概率P_x，softmax后的概率
    criterion_Py1 = nn.MSELoss()  # 移动维度起始点的概率P_y1，softmax后的概率
    criterion_Py2 = nn.MSELoss()  # 移动维度终点点的概率P_y2，softmax后的概率
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    train(model, train_loader, optimizer, epochs=10)