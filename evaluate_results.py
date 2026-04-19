# =============================================================================
# EVALUATION — Plant Disease Severity Grading
# 
# =============================================================================

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cpu')

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    'data_root':  r'C:\Users\DELL\Desktop\PlantVillage',
    'model_path': r'C:\Users\DELL\Desktop\PlantVillage\results\best_model.pth',
    'save_dir':   r'C:\Users\DELL\Desktop\PlantVillage\results',
    'img_size':   224,
    'batch_size': 32,   # Larger batch = faster evaluation (no gradients needed)
    'num_classes': 4,
}

SEVERITY_NAMES = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

SEVERITY_MAP = {
    'Tomato_healthy':                                0,
    'Potato___healthy':                              0,
    'Pepper__bell___healthy':                        0,
    'Tomato_Early_blight':                           1,
    'Tomato_Bacterial_spot':                         1,
    'Tomato_Septoria_leaf_spot':                     1,
    'Potato___Early_blight':                         1,
    'Pepper__bell___Bacterial_spot':                 1,
    'Tomato_Leaf_Mold':                              2,
    'Tomato_Spider_mites_Two_spotted_spider_mite':   2,
    'Tomato__Target_Spot':                           2,
    'Tomato_Late_blight':                            3,
    'Tomato__Tomato_YellowLeaf__Curl_Virus':         3,
    'Tomato__Tomato_mosaic_virus':                   3,
    'Potato___Late_blight':                          3,
}

# =============================================================================
# DATASET
# =============================================================================
class PlantSeverityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples, self.transform = [], transform
        for folder in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, folder)
            if not os.path.isdir(path) or folder not in SEVERITY_MAP:
                continue
            severity = SEVERITY_MAP[folder]
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(path, f), severity))
        print(f"  Loaded {len(self.samples)} images")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, severity = self.samples[idx]
        try:    img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (224, 224))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(severity, dtype=torch.long)

# =============================================================================
# MODEL + LOSS (must match training exactly)
# =============================================================================
class CORNLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks   = num_classes - 1

    def predict(self, logits):
        cum  = torch.sigmoid(logits)
        prob = torch.zeros(logits.size(0), self.num_classes)
        prob[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, self.num_classes - 1):
            prob[:, k] = cum[:, k-1] - cum[:, k]
        prob[:, -1] = cum[:, -1]
        prob = torch.clamp(prob, min=0)
        return torch.argmax(prob, dim=1), prob

class PlantSeverityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 3)
        )
    def forward(self, x): return self.backbone(x)

# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model():
    model = PlantSeverityModel()
    print(f"\n  Loading: {CONFIG['model_path']}")
    try:
        ckpt = torch.load(CONFIG['model_path'], map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Checkpoint: epoch {ckpt.get('epoch','?')} | val_acc {ckpt.get('val_acc','?'):.2f}%")
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"  ERROR: {e}")
        raise
    model.eval()
    return model

# =============================================================================
# EVALUATE
# =============================================================================
def evaluate(model, criterion, loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Evaluating', ncols=70):
            logits = model(images)
            preds, probs = criterion.predict(logits)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# =============================================================================
# PLOTS
# =============================================================================
def plot_confusion(labels, preds):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    names   = [SEVERITY_NAMES[i] for i in range(4)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title('Confusion Matrix — Counts', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('True Severity'); axes[0].set_xlabel('Predicted Severity')

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=names, yticklabels=names, ax=axes[1])
    axes[1].set_title('Confusion Matrix — Normalized', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('True Severity'); axes[1].set_xlabel('Predicted Severity')

    plt.suptitle('Ordinal ResNet18 — Test Set Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(CONFIG['save_dir'], 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confusion_matrix.png")


def plot_per_class_metrics(labels, preds):
    names = [SEVERITY_NAMES[i] for i in range(4)]
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall    = recall_score(labels, preds, average=None, zero_division=0)
    f1        = f1_score(labels, preds, average=None, zero_division=0)

    x = np.arange(4)
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w,   precision, w, label='Precision', color='steelblue',  alpha=0.85)
    ax.bar(x,       recall,    w, label='Recall',    color='darkorange', alpha=0.85)
    ax.bar(x + w,   f1,        w, label='F1-Score',  color='seagreen',   alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Score'); ax.set_xlabel('Severity Level')
    ax.set_title('Per-Class Precision, Recall, F1 — Test Set', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - w,   p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
        ax.text(i,       r + 0.02, f'{r:.2f}', ha='center', fontsize=9)
        ax.text(i + w,   f + 0.02, f'{f:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(CONFIG['save_dir'], 'per_class_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_metrics.png")


def plot_error_distribution(labels, preds):
    errors = np.abs(preds - labels)
    counts = [np.sum(errors == d) for d in range(4)]
    pcts   = [c / len(errors) * 100 for c in counts]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    labels_x = ['0 — Exact\n(correct)', '1 — One level\noff', '2 — Two levels\noff', '3 — Three levels\noff']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(range(4), counts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels_x, fontsize=10)
    axes[0].set_ylabel('Number of Samples'); axes[0].set_xlabel('Prediction Error Distance')
    axes[0].set_title('Error Distance Distribution\n(Ordinal Model)', fontweight='bold', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    for bar, cnt, pct in zip(bars, counts, pcts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{cnt}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    axes[1].pie(pcts, labels=labels_x, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
    axes[1].set_title('Error Distance — Proportional View', fontweight='bold', fontsize=12)

    plt.suptitle('Ordinal Model Error Analysis — Test Set', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(CONFIG['save_dir'], 'error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_distribution.png")


def plot_confidence_distribution(labels, preds, probs):
    correct_conf   = [probs[i][preds[i]] for i in range(len(preds)) if preds[i] == labels[i]]
    incorrect_conf = [probs[i][preds[i]] for i in range(len(preds)) if preds[i] != labels[i]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(correct_conf,   bins=30, alpha=0.7, color='seagreen',  label=f'Correct ({len(correct_conf)})')
    ax.hist(incorrect_conf, bins=30, alpha=0.7, color='tomato',    label=f'Incorrect ({len(incorrect_conf)})')
    ax.set_xlabel('Model Confidence (softmax probability)', fontsize=12)
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG['save_dir'], 'confidence_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confidence_distribution.png")


def plot_sample_predictions(model, criterion, test_set):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    indices = np.random.choice(len(test_set), 12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_t, true_lbl = test_set[idx]
            logits = model(img_t.unsqueeze(0))
            pred_lbl, probs = criterion.predict(logits)
            conf = probs[0][pred_lbl.item()].item() * 100
            img  = (img_t * std[:, None, None] + mean[:, None, None]).permute(1,2,0).clamp(0,1).numpy()
            axes[i].imshow(img)
            correct = true_lbl.item() == pred_lbl.item()
            axes[i].set_title(
                f'True: {SEVERITY_NAMES[true_lbl.item()]}\n'
                f'Pred: {SEVERITY_NAMES[pred_lbl.item()]} ({conf:.1f}%)',
                color='green' if correct else 'red', fontsize=9, fontweight='bold'
            )
            axes[i].axis('off')

    plt.suptitle('Sample Test Predictions  |  Green=Correct  Red=Wrong',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(CONFIG['save_dir'], 'sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sample_predictions.png")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*50)
    print("  Fast Evaluation — Plant Severity Model")
    print("="*50)

    # Load data (same seed = same test split as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_ds  = PlantSeverityDataset(CONFIG['data_root'], transform)
    total    = len(full_ds)
    train_sz = int(0.80 * total)
    val_sz   = int(0.10 * total)
    test_sz  = total - train_sz - val_sz

    _, _, test_set = torch.utils.data.random_split(
        full_ds, [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'],
                             shuffle=False, num_workers=0)
    print(f"  Test set: {len(test_set)} images")

    # Load model
    model     = load_model()
    criterion = CORNLoss()

    # Evaluate
    print("\nRunning test set evaluation...")
    preds, labels, probs = evaluate(model, criterion, test_loader)

    # Metrics
    accuracy = (preds == labels).mean() * 100
    mae      = np.abs(preds - labels).mean()
    qwk      = cohen_kappa_score(labels, preds, weights='quadratic')

    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  MAE      : {mae:.4f}  (0 = perfect)")
    print(f"  QWK      : {qwk:.4f}  (>0.60 = substantial)")
    print(f"{'='*50}")
    names = [SEVERITY_NAMES[i] for i in range(4)]
    print(classification_report(labels, preds, target_names=names, digits=3))

    # Save JSON
    results = {
        'test_accuracy': round(accuracy, 4),
        'test_mae':      round(float(mae), 4),
        'test_qwk':      round(float(qwk), 4),
        'per_class': classification_report(labels, preds,
                         target_names=names, output_dict=True)
    }
    json_path = os.path.join(CONFIG['save_dir'], 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: test_results.json")

    # Generate all plots
    print("\nGenerating plots...")
    plot_confusion(labels, preds)
    plot_per_class_metrics(labels, preds)
    plot_error_distribution(labels, preds)
    plot_confidence_distribution(labels, preds, probs)
    plot_sample_predictions(model, criterion, test_set)

    print(f"\n{'='*50}")
    print(f"  DONE — All files in results/")
    print(f"    confusion_matrix.png")
    print(f"    per_class_metrics.png")
    print(f"    error_distribution.png")
    print(f"    confidence_distribution.png")
    print(f"    sample_predictions.png")
    print(f"    test_results.json")
    print(f"{'='*50}")
    print(f"\n  PASTE THESE INTO YOUR THESIS:")
    print(f"    Accuracy : {accuracy:.2f}%")
    print(f"    MAE      : {mae:.4f}")
    print(f"    QWK      : {qwk:.4f}")

if __name__ == '__main__':
    main()