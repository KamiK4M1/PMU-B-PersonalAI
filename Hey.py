import torch
import torch.nn as nn

class NUInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1), nn.BatchNorm2d(mid), nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid), nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(mid), nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid), nn.ReLU()
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

class NUInNetMultiTask(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.block1 = NUInceptionBlock(32, 64)
        self.block2 = NUInceptionBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_fc = nn.Linear(128, 64)  # Shared layer

        self.classifier = nn.Linear(64, num_classes)
        self.regressor = nn.Linear(64, 1)  # Regression 0–100

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = nn.functional.relu(self.shared_fc(x))

        cls_logits = self.classifier(x)
        reg_score = torch.clamp(self.regressor(x).squeeze(1), 0, 100)  # Clamp to [0, 100]

        return cls_logits, reg_score

def multitask_loss(cls_logits, cls_targets, reg_preds, reg_targets, alpha=1.0, beta=0.01):
    cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_targets)
    reg_loss = nn.MSELoss()(reg_preds, reg_targets)
    return alpha * cls_loss + beta * reg_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class FoodDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        """
        image_dir: path to images
        annotations: list of tuples (filename, label, score)
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename, label, score = self.annotations[idx]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        image = self.transform(image)
        return image, (torch.tensor(label), torch.tensor(score, dtype=torch.float32))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# สมมุติใช้ NUInNetMultiTask จากที่เขียนไว้ด้านบน
model = NUInNetMultiTask(num_classes=50).to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def multitask_loss(class_logits, class_targets, score_preds, score_targets,
                   alpha=1.0, beta=0.01):
    cls_loss = nn.CrossEntropyLoss()(class_logits, class_targets)
    reg_loss = nn.MSELoss()(score_preds, score_targets)
    return alpha * cls_loss + beta * reg_loss

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for images, (labels, scores) in tqdm(dataloader):
        images, labels, scores = images.cuda(), labels.cuda(), scores.cuda()

        optimizer.zero_grad()
        cls_logits, reg_output = model(images)
        loss = multitask_loss(cls_logits, labels, reg_output, scores)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# สมมุติข้อมูล
annotations = [
    ('padthai1.jpg', 0, 82.3),
    ('tomyum1.jpg', 1, 91.0),
    ('green_curry.jpg', 2, 76.5),
    # ...
]

dataset = FoodDataset(image_dir='images/', annotations=annotations)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train loop
for epoch in range(20):
    avg_loss = train_epoch(model, dataloader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

def evaluate(model, dataloader):
    model.eval()
    total_acc, total_reg_loss, count = 0, 0, 0
    with torch.no_grad():
        for images, (labels, scores) in dataloader:
            images, labels, scores = images.cuda(), labels.cuda(), scores.cuda()
            cls_logits, reg_output = model(images)

            preds = cls_logits.argmax(dim=1)
            total_acc += (preds == labels).sum().item()

            reg_loss = F.mse_loss(reg_output, scores, reduction='sum')
            total_reg_loss += reg_loss.item()
            count += len(labels)
    acc = total_acc / count
    rmse = (total_reg_loss / count) ** 0.5
    return acc, rmse

from sklearn.metrics import f1_score, r2_score

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    all_score_preds = []

    with torch.no_grad():
        for images, (labels, scores) in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            scores = scores.cuda()

            cls_logits, reg_output = model(images)

            preds = torch.argmax(cls_logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            all_scores.append(scores.cpu())
            all_score_preds.append(reg_output.cpu())

    # รวมทั้งหมด
    y_true_cls = torch.cat(all_labels).numpy()
    y_pred_cls = torch.cat(all_preds).numpy()

    y_true_score = torch.cat(all_scores).numpy()
    y_pred_score = torch.cat(all_score_preds).numpy()

    # F1: classification (macro เหมาะกับ class ไม่บาลานซ์)
    f1 = f1_score(y_true_cls, y_pred_cls, average='macro')

    # R²: regression
    r2 = r2_score(y_true_score, y_pred_score)

    return f1, r2

best_f1 = 0
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    f1, r2 = evaluate(model, val_loader)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_model.pt')
    
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | F1={f1:.4f} | R2={r2:.4f}")
