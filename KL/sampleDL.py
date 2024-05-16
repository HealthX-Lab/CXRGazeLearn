import torch
import torch.nn as nn
import torch.optim as optim

# 假设的特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.network(x)

# 对比损失
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negatives):
        # 计算anchor和positive的相似度
        pos_sim = self.cosine_similarity(anchor, positive).unsqueeze(-1)
        
        # 正确扩展anchor以匹配negatives的形状
        # 假设negatives的形状为 [batch_size, num_negatives, feature_size]
        # 我们需要将anchor从 [batch_size, feature_size] 扩展为 [batch_size, 1, feature_size]
        # 这样每个anchor都可以与它的每个负例比较
        anchor_expanded = anchor.unsqueeze(1)
        
        # 现在我们可以计算每个anchor与其所有负例的相似度
        # 这里不需要在anchor上调用unsqueeze(0)，因为我们已经通过unsqueeze(1)调整了形状
        neg_sim = self.cosine_similarity(anchor_expanded, negatives)
        
        # 应用温度参数
        pos_sim /= self.temperature
        neg_sim /= self.temperature
        
        # 计算损失
        losses = -pos_sim + torch.log(torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=-1))
        return losses.mean()


# 初始化模型和优化器
model = FeatureExtractor()
loss_fn = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
num_epochs = 10  # 训练周期数
batch_size = 32  # 批量大小

# 模拟训练循环
for epoch in range(num_epochs):
    # 假设有一个数据加载器，每次迭代提供数据
    for batch_idx in range(100):  # 假设每个epoch有100个batch
        # 生成模拟数据
        anchor = torch.randn(batch_size, 784)
        positive = torch.randn(batch_size, 784)
        negatives = torch.randn(batch_size, 5, 784)

        # 前向传播
        anchor_features = model(anchor)
        positive_features = model(positive)
        negatives_features = model(negatives)

        # 计算损失
        loss = loss_fn(anchor_features, positive_features, negatives_features)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
