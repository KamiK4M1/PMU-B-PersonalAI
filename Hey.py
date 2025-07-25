import torch
import torch.nn as nn

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            DepthWiseSeperable(32, 64, 1),
            DepthWiseSeperable(64, 128, 2),
            DepthWiseSeperable(128, 128, 1),
            DepthWiseSeperable(128, 256, 2),
            DepthWiseSeperable(256, 256, 1),
            DepthWiseSeperable(256, 512, 2),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 1024, 2),
            DepthWiseSeperable(1024, 1024, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Linear(1024, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = nn.functional.relu(self.shared_fc(x))

        cls_logits = self.classifier(x)
        reg_score = self.regressor(x).squeeze(1)  # if your targets are already 0–100 float

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

    class_names = ['ข้าวผัด', 'ต้มยำ', 'ผัดไทย', ...]  # 50 รายการ

print(f"คาดว่าเป็นเมนู: {class_names[class_idx]}")
print(f"ให้คะแนน: {score:.2f} / 100")
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | F1={f1:.4f} | R2={r2:.4f}")


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # ปรับตามที่ใช้เทรน
])

pred_classes = []
pred_scores = []

model.eval()
with torch.no_grad():
    for img_path in df["image_path"]:
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        cls_logits, reg_score = model(img)

        class_idx = torch.argmax(cls_logits, dim=1).item()
        score = reg_score.item() * 100  # เพราะโมเดล clamp ไว้ 0–1

        pred_classes.append(class_idx)
        pred_scores.append(score)

df["pred_class"] = pred_classes
df["pred_score"] = pred_scores
