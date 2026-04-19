# =============================================================================
# Plant Disease Severity Grading Using Ordinal Regression
# Dataset: PlantVillage (20,638 images → 4 severity levels)
# Author: MS Data Science Thesis
# Requirements: torch, torchvision, tqdm, matplotlib, seaborn, scikit-learn
# Install: pip install torch torchvision tqdm matplotlib seaborn scikit-learn
# Runtime: ~4-6 hours on CPU (i5, 16GB RAM)
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Set random seed for reproducibility ───────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device (auto-detects GPU, falls back to CPU) ──────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cpu':
    print("Note: Running on CPU. Training will take 4-6 hours. Reduce EPOCHS to 5 for faster testing.")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

CONFIG = {
    'data_root':    '.',   # Current folder
    'batch_size':   16,                 # Reduced for CPU (use 32 if you have GPU)
    'epochs':       10,                 # Reduce to 3-5 for quick testing
    'lr':           1e-4,
    'weight_decay': 1e-4,
    'img_size':     224,
    'num_classes':  4,                  # 0=Healthy, 1=Mild, 2=Moderate, 3=Severe
    'num_workers':  0,                  # 0 = safer on Windows
    'save_dir':     './results',
    'model_name':   'resnet18',         # resnet18 (faster) or resnet34 (more accurate)
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

SEVERITY_NAMES = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}


# =============================================================================
# SECTION 2: SEVERITY MAPPING (Exact PlantVillage folder names)
# =============================================================================

SEVERITY_MAP = {
    # Severity 0 — Healthy
    'Tomato_healthy':                                       0,
    'Potato___healthy':                                     0,
    'Pepper__bell___healthy':                               0,

    # Severity 1 — Mild (early-stage, localized)
    'Tomato_Early_blight':                                  1,
    'Tomato_Bacterial_spot':                                1,
    'Tomato_Septoria_leaf_spot':                            1,
    'Potato___Early_blight':                                1,
    'Pepper__bell___Bacterial_spot':                        1,

    # Severity 2 — Moderate (progressive, treatment needed)
    'Tomato_Leaf_Mold':                                     2,
    'Tomato_Spider_mites_Two_spotted_spider_mite':          2,
    'Tomato__Target_Spot':                                  2,

    # Severity 3 — Severe (critical, systemic)
    'Tomato_Late_blight':                                   3,
    'Tomato__Tomato_YellowLeaf__Curl_Virus':                3,
    'Tomato__Tomato_mosaic_virus':                          3,
    'Potato___Late_blight':                                 3,
}


# =============================================================================
# SECTION 3: DATASET CLASS
# =============================================================================

class PlantSeverityDataset(Dataset):
    """
    Custom Dataset for PlantVillage severity grading.
    Maps 15 disease folders → 4 ordinal severity levels.
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        skipped = []

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            if folder_name not in SEVERITY_MAP:
                skipped.append(folder_name)
                continue

            severity = SEVERITY_MAP[folder_name]

            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(folder_path, img_file), severity)
                    )

        print(f"\n{'='*55}")
        print(f"  Dataset loaded: {len(self.samples)} images")
        if skipped:
            print(f"  Skipped folders (not in map): {skipped}")
        self._print_distribution()

    def _print_distribution(self):
        labels = [s[1] for s in self.samples]
        dist = Counter(labels)
        total = len(labels)
        print(f"\n  Severity Distribution:")
        for k in sorted(dist):
            bar = '█' * int(dist[k] / total * 30)
            print(f"  Level {k} {SEVERITY_NAMES[k]:10s}: {dist[k]:5d}  {bar}")
        print(f"{'='*55}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, severity = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 128, 0))

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(severity, dtype=torch.long)


# =============================================================================
# SECTION 4: DATA LOADING
# =============================================================================

def get_dataloaders(config):
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = PlantSeverityDataset(config['data_root'], transform=train_transform)

    total     = len(full_dataset)
    train_sz  = int(0.80 * total)
    val_sz    = int(0.10 * total)
    test_sz   = total - train_sz - val_sz

    train_set, val_set, test_set = random_split(
        full_dataset, [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply val/test transforms (no augmentation)
    val_set.dataset.transform  = val_transform
    test_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=config['num_workers'],
                              pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])
    test_loader  = DataLoader(test_set,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])

    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 5: ORDINAL REGRESSION LOSS (CORN)
# =============================================================================
#
# WHY ORDINAL LOSS INSTEAD OF CROSS-ENTROPY?
# ------------------------------------------
# Cross-entropy treats all misclassifications equally:
#   Predicting "Moderate" when truth is "Healthy" = same penalty as
#   Predicting "Mild"     when truth is "Healthy" — WRONG for grading!
#
# Ordinal loss respects the order: Healthy < Mild < Moderate < Severe
#   A prediction that is 1 level off costs less than 2 levels off.
#
# CORN (Cumulative Odds Regression for Neural Networks):
#   - Decomposes K-class ordinal problem into K-1 binary tasks
#   - Task 1: P(y > 0) = P(not Healthy)
#   - Task 2: P(y > 1) = P(Moderate or Severe)
#   - Task 3: P(y > 2) = P(Severe)
#   - Final prediction: argmax of reconstructed probabilities
# =============================================================================

class CORNLoss(nn.Module):
    """
    CORN: Cumulative Odds Regression for Neural Networks
    Reference: Shi et al. (2023) - Revisiting ordinal regression neural networks
    
    For 4 classes (0,1,2,3), creates 3 binary classifiers:
      head 0: P(y > 0)  — Is it worse than Healthy?
      head 1: P(y > 1)  — Is it worse than Mild?
      head 2: P(y > 2)  — Is it worse than Moderate?
    """
    def __init__(self, num_classes=4):
        super(CORNLoss, self).__init__()
        self.num_classes = num_classes
        self.num_tasks   = num_classes - 1   # 3 binary tasks

    def forward(self, logits, targets):
        """
        logits:  (batch, num_tasks) — raw scores for each binary threshold
        targets: (batch,)           — ordinal class labels 0..K-1
        """
        # Convert to binary labels for each threshold task
        # For task k: label=1 if target > k, else 0
        binary_targets = torch.zeros(
            targets.size(0), self.num_tasks, device=targets.device
        )
        for k in range(self.num_tasks):
            binary_targets[:, k] = (targets > k).float()

        # Binary cross-entropy for each threshold task
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction='mean'
        )
        return loss

    def predict(self, logits):
        """
        Convert CORN logits → class predictions.
        P(y=k) reconstructed from cumulative probabilities.
        """
        # Cumulative probabilities: P(y > k) for k=0,1,2
        cum_probs = torch.sigmoid(logits)           # (batch, 3)

        batch_size = logits.size(0)
        class_probs = torch.zeros(
            batch_size, self.num_classes, device=logits.device
        )

        # P(y=0) = 1 - P(y>0)
        class_probs[:, 0] = 1.0 - cum_probs[:, 0]

        # P(y=k) = P(y>k-1) - P(y>k)  for k=1,2
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]

        # P(y=K-1) = P(y>K-2)
        class_probs[:, -1] = cum_probs[:, -1]

        # Clamp to avoid numerical negatives
        class_probs = torch.clamp(class_probs, min=0)

        return torch.argmax(class_probs, dim=1), class_probs


# =============================================================================
# SECTION 6: MODEL ARCHITECTURE
# =============================================================================
#
# WHY PRETRAINED RESNET?
# ----------------------
# Training from scratch on 20k images risks overfitting.
# ResNet18 pretrained on ImageNet already knows:
#   - Edge detection, texture patterns, color gradients
#   - These features transfer well to leaf disease patterns
# We only retrain the final layer for our ordinal task.
#
# ARCHITECTURE MODIFICATION:
#   Standard ResNet18 final layer: Linear(512 → 1000) [ImageNet classes]
#   Our modification:              Linear(512 → 3)    [3 CORN binary tasks]
# =============================================================================

class PlantSeverityModel(nn.Module):
    """
    ResNet18/34 backbone with CORN ordinal output head.
    Final layer outputs 3 logits (for K-1=3 binary thresholds).
    """
    def __init__(self, backbone='resnet18', num_classes=4, pretrained=True):
        super(PlantSeverityModel, self).__init__()
        self.num_tasks = num_classes - 1

        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace final classification layer
        # Original: Linear(512 → 1000)
        # Ours:     Linear(512 → 3) for CORN binary tasks
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),                      # Regularization
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_tasks)          # 3 binary threshold outputs
        )

        print(f"\nModel: {backbone} + CORN ordinal head")
        print(f"Backbone parameters: {sum(p.numel() for p in self.backbone.parameters()):,}")

    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# SECTION 7: EVALUATION METRICS
# =============================================================================

def compute_metrics(all_preds, all_labels, split_name=''):
    """
    Compute ordinal-appropriate metrics:
    - Accuracy: standard correct predictions
    - MAE: Mean Absolute Error (key for ordinal — penalizes distant errors more)
    - QWK: Quadratic Weighted Kappa (gold standard for ordinal agreement)
    - Per-class precision/recall
    """
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    mae      = np.abs(all_preds - all_labels).mean()
    qwk      = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"\n{'─'*45}")
    print(f"  {split_name} Results:")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  MAE      : {mae:.4f}  (lower=better; 0=perfect)")
    print(f"  QWK      : {qwk:.4f}  (>0.60 = substantial agreement)")
    print(f"{'─'*45}")

    # Per-class report
    target_names = [SEVERITY_NAMES[i] for i in range(4)]
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, digits=3))

    return {'accuracy': accuracy, 'mae': mae, 'qwk': qwk}


# =============================================================================
# SECTION 8: TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc='  Training', leave=False, ncols=80)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)                          # (batch, 3)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds, _ = criterion.predict(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Evaluating', leave=False, ncols=80):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds, _ = criterion.predict(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return avg_loss, accuracy, all_preds, all_labels


def train_model(model, train_loader, val_loader, config, device):
    criterion = CORNLoss(num_classes=config['num_classes'])
    optimizer = optim.Adam(model.parameters(),
                           lr=config['lr'],
                           weight_decay=config['weight_decay'])

    # Learning rate scheduler: reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_loss = float('inf')
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')

    print(f"\n{'='*55}")
    print(f"  Starting Training: {config['epochs']} epochs")
    print(f"  Batch size: {config['batch_size']} | LR: {config['lr']}")
    print(f"{'='*55}")

    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch [{epoch}/{config['epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = (time.time() - start_time) / 60
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.2f}%")
        print(f"  Elapsed: {elapsed:.1f} min")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

    print(f"\nTotal training time: {(time.time()-start_time)/60:.1f} minutes")
    return history, best_model_path, criterion


# =============================================================================
# SECTION 9: VISUALIZATION
# =============================================================================

def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val Loss',   markersize=4)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CORN Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val Acc',   markersize=4)
    axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(all_labels, all_preds, save_dir, title='Confusion Matrix'):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [SEVERITY_NAMES[i] for i in range(4)]

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'{title} (Normalized)', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_predictions(model, criterion, dataset_subset, device, save_dir, n=12):
    """Show sample predictions with true vs predicted severity."""
    model.eval()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset_subset), min(n, len(dataset_subset)), replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, true_label = dataset_subset[idx]
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred_label, class_probs = criterion.predict(logits)
            confidence = class_probs[0][pred_label.item()].item() * 100

            # Denormalize for display
            mean = torch.tensor([0.485, 0.456, 0.406])
            std  = torch.tensor([0.229, 0.224, 0.225])
            img_display = img_tensor * std[:, None, None] + mean[:, None, None]
            img_display = img_display.permute(1, 2, 0).clamp(0, 1).numpy()

            axes[i].imshow(img_display)
            true_name = SEVERITY_NAMES[true_label.item()]
            pred_name = SEVERITY_NAMES[pred_label.item()]
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(
                f'True: {true_name}\nPred: {pred_name} ({confidence:.1f}%)',
                color=color, fontsize=9
            )
            axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 10: BASELINE COMPARISON (Cross-entropy vs Ordinal)
# =============================================================================

class BaselineModel(nn.Module):
    """Standard cross-entropy baseline for comparison."""
    def __init__(self, num_classes=4, pretrained=True):
        super(BaselineModel, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)   # Standard 4-class output
        )

    def forward(self, x):
        return self.backbone(x)


def train_baseline(train_loader, val_loader, config, device):
    """Train cross-entropy baseline for comparison."""
    print("\n" + "="*55)
    print("  Training BASELINE (Cross-Entropy)")
    print("="*55)

    model     = BaselineModel(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    best_path = os.path.join(config['save_dir'], 'baseline_model.pth')

    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Baseline Epoch {epoch}',
                                   leave=False, ncols=80):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total * 100

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total * 100
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    return model, best_path


# =============================================================================
# SECTION 11: EXPERIMENT RUNNER
# =============================================================================

def run_comparison_experiment(test_loader, ordinal_model, baseline_model,
                               ordinal_criterion, device, save_dir):
    """Compare ordinal vs cross-entropy baseline on test set."""
    print("\n" + "="*55)
    print("  EXPERIMENT: Ordinal vs Cross-Entropy Comparison")
    print("="*55)

    results = {}

    # Evaluate ordinal model
    ordinal_model.eval()
    ord_preds, ord_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Ordinal', leave=False):
            images = images.to(device)
            logits = ordinal_model(images)
            preds, _ = ordinal_criterion.predict(logits)
            ord_preds.extend(preds.cpu().numpy())
            ord_labels.extend(labels.numpy())

    results['ordinal'] = compute_metrics(ord_preds, ord_labels, 'ORDINAL (CORN)')

    # Evaluate baseline model
    baseline_model.eval()
    base_preds, base_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Baseline', leave=False):
            images = images.to(device)
            outputs = baseline_model(images)
            preds = outputs.argmax(dim=1)
            base_preds.extend(preds.cpu().numpy())
            base_labels.extend(labels.numpy())

    results['baseline'] = compute_metrics(base_preds, base_labels, 'BASELINE (Cross-Entropy)')

    # Summary table
    print("\n" + "="*55)
    print("  COMPARISON SUMMARY")
    print(f"  {'Metric':<12} {'Ordinal':>12} {'Baseline':>12} {'Winner':>10}")
    print("  " + "-"*46)
    for metric in ['accuracy', 'mae', 'qwk']:
        ord_val  = results['ordinal'][metric]
        base_val = results['baseline'][metric]
        if metric == 'mae':
            winner = 'Ordinal' if ord_val < base_val else 'Baseline'
        else:
            winner = 'Ordinal' if ord_val > base_val else 'Baseline'
        print(f"  {metric.upper():<12} {ord_val:>12.4f} {base_val:>12.4f} {winner:>10}")
    print("="*55)

    # Confusion matrices for both
    plot_confusion_matrix(ord_labels, ord_preds, save_dir, 'Ordinal Model')

    # Save results
    import json
    with open(os.path.join(save_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# SECTION 12: MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "="*55)
    print("  Plant Disease Severity Grading")
    print("  Ordinal Regression with CORN Loss")
    print("="*55)

    # ── Load data ─────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG)

    # ── Build ordinal model ───────────────────────────────────
    ordinal_model = PlantSeverityModel(
        backbone=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        pretrained=True
    ).to(DEVICE)

    # ── Train ordinal model ───────────────────────────────────
    history, best_path, criterion = train_model(
        ordinal_model, train_loader, val_loader, CONFIG, DEVICE
    )

    # ── Load best checkpoint ──────────────────────────────────
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    ordinal_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    # ── Final evaluation on test set ──────────────────────────
    print("\n" + "="*55)
    print("  FINAL TEST SET EVALUATION")
    _, _, test_preds, test_labels = evaluate(
        ordinal_model, test_loader, criterion, DEVICE
    )
    test_metrics = compute_metrics(test_preds, test_labels, 'TEST SET')

    # ── Visualizations ────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_training_curves(history, CONFIG['save_dir'])
    plot_confusion_matrix(test_labels, test_preds, CONFIG['save_dir'])

    # Sample predictions from test set
    _, _, test_set_raw = get_dataloaders(CONFIG)
    visualize_predictions(
        ordinal_model, criterion,
        test_loader.dataset, DEVICE, CONFIG['save_dir']
    )

    # ── Baseline comparison ───────────────────────────────────
    print("\nTraining baseline for comparison...")
    baseline_model, _ = train_baseline(train_loader, val_loader, CONFIG, DEVICE)
    run_comparison_experiment(
        test_loader, ordinal_model, baseline_model,
        criterion, DEVICE, CONFIG['save_dir']
    )

    print("\n" + "="*55)
    print("  Training Complete!")
    print(f"  Results saved to: {CONFIG['save_dir']}/")
    print(f"  Files:")
    print(f"    best_model.pth          — Best ordinal model weights")
    print(f"    training_curves.png     — Loss & accuracy plots")
    print(f"    confusion_matrix.png    — Error analysis")
    print(f"    sample_predictions.png  — Visual predictions")
    print(f"    comparison_results.json — Ordinal vs baseline metrics")
    print("="*55)


if __name__ == '__main__':
    main()


# =============================================================================
# QUICK TEST (Run this first to verify setup before full training)
# =============================================================================
# Uncomment and run this block to test in ~2 minutes:
#
CONFIG['epochs']     = 2
CONFIG['batch_size'] = 8
main()
