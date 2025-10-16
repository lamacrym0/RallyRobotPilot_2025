# pip install torch --upgrade  (si besoin)
import os, glob, pickle, lzma
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# 0) Sélection des fichiers à charger (plusieurs fichiers)
#    Tu peux mettre:
#      - une liste: ["record_8.xz", "record_9.xz"]
#      - un motif glob: "data/*.xz"
#      - un dossier: "data_logs/"
#    -> tous seront résolus en une liste de fichiers
# ------------------------------------------------------------
INPUT_FILES = ["record_0.npz","record_alex_0n.npz","record_alex_1n.npz","record_alex_2n.npz","record_alex_3n.npz","record_alex_4n.npz","record_alex_5n.npz","record_alex_6n.npz","record_alex_7n.npz"]


def gather_paths(specs):
    """Résout specs (str/Path ou liste) en liste de fichiers existants."""
    if isinstance(specs, (str, Path)):
        specs = [specs]
    files = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            # Tous les fichiers du dossier (récursif)
            files += [str(fp) for fp in p.rglob("*") if fp.is_file()]
        else:
            # Motif glob OU fichier direct
            matches = glob.glob(str(p))
            if matches:
                files += [str(fp) for fp in matches if Path(fp).is_file()]
            elif p.exists() and p.is_file():
                files.append(str(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour {specs}")
    return files


# ------------------------------------------------------------
# 1) Chargement & parsing -> X:[N,16], y:[N,4] (schéma A uniquement)
#    Chaque item: (speed: float, raycast_distances: tuple[15], current_controls: tuple[4])
# ------------------------------------------------------------
def _pickle_load_any(path):
    """Tente d'abord lzma, sinon open standard (pour fichiers non-compressés)."""
    try:
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    except (lzma.LZMAError, EOFError, OSError):
        with open(path, "rb") as f:
            return pickle.load(f)

def load_raw(path, strict=True):
    """Charge un fichier et retourne une liste de tuples (speed, dists[15], ctrls[4])."""
    try:
        data = _pickle_load_any(path)
    except Exception as e:
        if strict:
            raise
        print(f"⚠️  Fichier ignoré (lecture impossible): {path} -> {e}")
        return []

    samples = []
    for e in data:
        try:
            speed = float(e.car_speed)
            dists = tuple(float(v) for v in e.raycast_distances)
            ctrls = tuple(float(v) for v in e.current_controls)
            if len(dists) != 15:
                raise ValueError(f"raycast_distances doit avoir 15 valeurs, trouvé {len(dists)}")
            if len(ctrls) != 4:
                raise ValueError(f"current_controls doit avoir 4 valeurs, trouvé {len(ctrls)}")
            samples.append((speed, dists, ctrls))
        except Exception as ex:
            if strict:
                raise
            # On ignore juste cet enregistrement corrompu
            print(f"⚠️  Sample ignoré dans {path}: {ex}")
    return samples

# --> Charge plusieurs fichiers
file_list = gather_paths(INPUT_FILES)
print(f"Fichiers détectés ({len(file_list)}):")
for p in file_list:
    print(" -", p)

raw_samples_all = []
for p in file_list:
    cur = load_raw(p, strict=False)  # passe en True si tu veux être strict
    print(f"  {p}: {len(cur)} échantillons")
    raw_samples_all.extend(cur)

if not raw_samples_all:
    raise RuntimeError("Aucun échantillon valide après chargement de tous les fichiers.")

# Concatène en X:[N,16], y:[N,4]
X_list, y_list = [], []
for speed, dists, ctrls in raw_samples_all:
    X_list.append([speed] + list(dists))  # 1 + 15 = 16 features
    y_list.append(list(ctrls))            # 4 labels 0/1

X = torch.tensor(X_list, dtype=torch.float32)  # [N,16]
y = torch.tensor(y_list, dtype=torch.float32)  # [N,4]
N = X.shape[0]
assert X.shape[1] == 16 and y.shape[1] == 4, (X.shape, y.shape)
print(f"Total échantillons: {N}")

# ------------------------------------------------------------
# 2) Split (train/val) + normalisation (stats calculées sur le train)
# ------------------------------------------------------------
val_ratio = 0.2 if N >= 5 else 0.0
n_val = int(N * val_ratio)
n_train = N - n_val

g = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=g)
train_idx = perm[:n_train]
val_idx   = perm[n_train:] if n_val > 0 else None

mean = X[train_idx].mean(0, keepdim=True)
std  = X[train_idx].std(0, keepdim=True).clamp_min(1e-6)
Xn   = (X - mean) / std

from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(Xn[train_idx], y[train_idx])
val_ds   = TensorDataset(Xn[val_idx],   y[val_idx]) if n_val > 0 else None

batch_size = min(64, max(4, n_train))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

# ------------------------------------------------------------
# 3) Modèle (multi-label 4 sorties)
# ------------------------------------------------------------
class ControllerMLP(nn.Module):
    def __init__(self, in_dim=16, hidden=64, out_dim=4, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim)  # logits (pas de sigmoid ici)
        )
    def forward(self, x): 
        return self.net(x)

model = ControllerMLP(16, 128, 4, p_drop=0.1)

# ------------------------------------------------------------
# 4) Loss (BCEWithLogits + pos_weight) & optimiseur
# ------------------------------------------------------------
with torch.no_grad():
    pos = y.sum(0)                  # [4] nb de 1 par contrôle
    neg = y.shape[0] - pos
    pos_weight = torch.where(pos > 0, neg / pos, torch.ones_like(pos))  # évite inf si aucune occurrence
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ------------------------------------------------------------
# 5) Entraînement / Évaluation
# ------------------------------------------------------------
def evaluate(loader):
    model.eval()
    total_loss, n_batches, elem_acc = 0.0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_loss += loss.item()
            elem_acc += (preds == yb).float().mean().item()
            n_batches += 1
    return total_loss / n_batches, elem_acc / n_batches

epochs = 40
for epoch in range(1, epochs + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    tr_loss, tr_acc = evaluate(train_loader)
    if val_loader:
        va_loss, va_acc = evaluate(val_loader)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")
    else:
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f}")

# ------------------------------------------------------------
# 6) Inférence (probas + seuil 0.5)
# ------------------------------------------------------------
model.eval()
def predict_one(speed, raycast_distances):
    assert len(raycast_distances) == 15
    x = torch.tensor([[float(speed)] + list(map(float, raycast_distances))], dtype=torch.float32)
    x = (x - mean) / std
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > 0.5).astype(int)
    return probs, preds  # [4], [4]

# ------------------------------------------------------------
# 7) Sauvegarde (avec normalisation)
# ------------------------------------------------------------
torch.save({
    "state_dict": model.state_dict(),
    "mean": mean.cpu(),
    "std": std.cpu(),
}, "controller_multilabel.pt")
print("Modèle sauvegardé -> controller_multilabel.pt")
