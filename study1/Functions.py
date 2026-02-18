# 
from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as sgn
from scipy.io import loadmat
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



#--------------------------------------------Preprocessing step-----------------------------------------------------------
def remove_baseline_filter(sample_rate: int):
    """Return SOS for baseline removal high-pass filter (elliptic)."""
    fc = 0.8
    fst = 0.2
    rp = 0.5
    rs = 40

    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)
    order, _ = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(order, wn, rp, rs, btype="high", ftype="ellip", output="sos")
    return sos


def preprocess_one_file(
    mat_path: Path,
    *,
    fs_in: int,
    fs_out: int,
    win_len: int,
    stride: int,
    keep_signals: List[str],
    scale: Dict[str, float],
    sos,
) -> np.ndarray:
    """
    One file -> (n_windows, len(keep_signals), win_len)
    """
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    rs = m["resting_state"]

    # baseline removal (filter at original sampling rate)
    ecg_f   = sgn.sosfiltfilt(sos, rs.ECG)
    fp_f    = sgn.sosfiltfilt(sos, rs.Fingerpulse)
    can_f   = sgn.sosfiltfilt(sos, rs.Cannula)
    therm_f = sgn.sosfiltfilt(sos, rs.Thermopod)

    # resample to fs_out
    ecg_60   = sgn.resample_poly(ecg_f,   up=fs_out, down=fs_in)
    fp_60    = sgn.resample_poly(fp_f,    up=fs_out, down=fs_in)
    can_60   = sgn.resample_poly(can_f,   up=fs_out, down=fs_in)
    therm_60 = sgn.resample_poly(therm_f, up=fs_out, down=fs_in)

    # scale
    sig = {
        "ECG_mV":       ecg_60   * scale["ECG_mV"],
        "Fingerpulse":  fp_60    * scale["Fingerpulse"],
        "Cannula":      can_60   * scale["Cannula"],
        "Thermopod":    therm_60 * scale["Thermopod"],
    }

    # windowing
    L = len(sig["ECG_mV"])
    starts = np.arange(0, L - win_len + 1, stride)

    n_windows = len(starts)
    n_ch = len(keep_signals)
    tensor = np.zeros((n_windows, n_ch, win_len), dtype=np.float32)

    for i, s in enumerate(starts):
        for j, name in enumerate(keep_signals):
            tensor[i, j, :] = sig[name][s : s + win_len]

    return tensor


def stem_to_subject_id(stem: str) -> int:
    m = re.search(r"participant_(\d+)", stem)
    if not m:
        raise ValueError(f"Could not parse participant id from: {stem}")
    return int(m.group(1))



def split_participants_by_pid(
    participant_files: List[Path],
    *,
    nose_range: Tuple[int, int] = (10, 112),
    mouth_range: Tuple[int, int] = (200, 312),
) -> Tuple[List[Path], List[Path], List[str]]:
    """Split files into nose vs mouth using participant_<id> in filename."""
    nose_files, mouth_files, skipped = [], [], []

    for f in participant_files:
        m = re.search(r"participant_(\d+)", f.stem)
        if not m:
            skipped.append(f.name)
            continue

        pid = int(m.group(1))

        if nose_range[0] <= pid <= nose_range[1]:
            nose_files.append(f)
        elif mouth_range[0] <= pid <= mouth_range[1]:
            mouth_files.append(f)
        else:
            skipped.append(f.name)

    return nose_files, mouth_files, skipped


def preprocess_group(
    files: List[Path],
    *,
    fs_in: int,
    fs_out: int,
    win_len: int,
    stride: int,
    keep_signals: List[str],
    scale: Dict[str, float],
    sos,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """Return dict stem->tensor and list of (filename, error)."""
    out: Dict[str, np.ndarray] = {}
    failed: List[Tuple[str, str]] = []

    for f in files:
        try:
            out[f.stem] = preprocess_one_file(
                f,
                fs_in=fs_in,
                fs_out=fs_out,
                win_len=win_len,
                stride=stride,
                keep_signals=keep_signals,
                scale=scale,
                sos=sos,
            )
        except Exception as e:
            failed.append((f.name, str(e)))

    return out, failed


def stack_by_sorted_keys(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Stack dict values into X with stable ordering."""
    keys = sorted(d.keys())
    X = np.stack([d[k] for k in keys], axis=0)
    return X, keys


#------------------------------------Plot funcion 

def training_curve_plot(
    title,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    batch_size=None,
    learning_rate=None,
    training_time=None,
    baseline_acc=None 
):
    lg, md, sm = 18, 13, 9

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.15, fontsize=lg)

    info = []
    if batch_size is not None:
        info.append(f"Batch size: {batch_size}")
    if learning_rate is not None:
        info.append(f"Learning rate: {learning_rate}")
    if training_time is not None:
        mins, secs = divmod(training_time, 60)
        info.append(f"Training time: {int(mins)} min {secs:.0f} sec")
    info.append(f"Epochs: {len(train_losses)}")
    fig.text(0.5, 0.99, " | ".join(info), ha='center', fontsize=md)

    x = range(1, len(train_losses) + 1)

    axs[0].plot(x, train_losses, label=f"Final train loss: {train_losses[-1]:.4f}")
    axs[0].plot(x, val_losses, label=f"Final val loss: {val_losses[-1]:.4f}")
    axs[0].set_title("Loss", fontsize=md)
    axs[0].set_xlabel("Epochs", fontsize=md)
    axs[0].set_ylabel("Loss", fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)

    train_accs = np.array(train_accs) * 100
    val_accs   = np.array(val_accs) * 100

    axs[1].plot(x, train_accs, label=f"Final train acc: {train_accs[-1]:.1f}%")
    axs[1].plot(x, val_accs, label=f"Final val acc: {val_accs[-1]:.1f}%")

    if baseline_acc is not None:
        axs[1].axhline(baseline_acc * 100, linestyle="--",
                       label=f"Baseline acc: {baseline_acc*100:.1f}%")

    axs[1].set_title("Accuracy", fontsize=md)
    axs[1].set_xlabel("Epochs", fontsize=md)
    axs[1].set_ylabel("Accuracy (%)", fontsize=md)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)

    plt.tight_layout()
    plt.show()



# ----------------------------------------- CNN modle--------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CNN1D(nn.Module):
    def __init__(self, in_ch: int, p_drop: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.classifier(x).view(-1)
    

# -----------------------------------split and loaders---------------------------

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

def prepare_split_data_loaders(
    X_mouth, y_mouth,
    test_size=0.2,
    val_size=0.2,          # fraction of trainval -> validation
    seed=6,                # ONE seed only
    batch_size=64,
    normalize=True,
    verbose=True,
    return_meta=True,
):
    P, W, C, L = X_mouth.shape
    idx = np.arange(P)

    if verbose:
        print("X_mouth:".ljust(12), X_mouth.shape)
        print("y_mouth:".ljust(12), y_mouth.shape,
              "  unique:", np.unique(y_mouth),
              "  counts:", np.bincount(y_mouth.astype(int)))

    # 1) split test vs train+val
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_mouth
    )

    # 2) split train vs val (deterministic from same seed)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_size,
        random_state=seed + 1,          # derived, still reproducible
        stratify=y_mouth[trainval_idx]
    )

    # participant-level splits
    X_train_p, y_train_p = X_mouth[train_idx], y_mouth[train_idx]
    X_val_p,   y_val_p   = X_mouth[val_idx],   y_mouth[val_idx]
    X_test_p,  y_test_p  = X_mouth[test_idx],  y_mouth[test_idx]

    mu = sd = None
    if normalize:
        mu = X_train_p.mean(axis=(0, 1, 3), keepdims=True)
        sd = X_train_p.std(axis=(0, 1, 3), keepdims=True) + 1e-8

        X_train_p = (X_train_p - mu) / sd
        X_val_p   = (X_val_p   - mu) / sd
        X_test_p  = (X_test_p  - mu) / sd

        if verbose:
            print("\nAfter normalization (train mean per ch):", X_train_p.mean(axis=(0,1,3)))
            print("After normalization (train std  per ch):", X_train_p.std(axis=(0,1,3)))

    # window-level
    X_train = X_train_p.reshape(len(train_idx) * W, C, L)
    y_train = np.repeat(y_train_p, W)

    X_val   = X_val_p.reshape(len(val_idx) * W, C, L)
    y_val   = np.repeat(y_val_p, W)

    X_test  = X_test_p.reshape(len(test_idx) * W, C, L)
    y_test  = np.repeat(y_test_p, W)

    if verbose:
        print("\nWindow-level shapes:")
        print("X_train:".ljust(12), X_train.shape, " y_train:", y_train.shape,
              " counts:", np.bincount(y_train.astype(int)))
        print("X_val  :".ljust(12), X_val.shape,   " y_val  :", y_val.shape,
              " counts:", np.bincount(y_val.astype(int)))
        print("X_test :".ljust(12), X_test.shape,  " y_test :", y_test.shape,
              " counts:", np.bincount(y_test.astype(int)))

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(WindowDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    if not return_meta:
        return train_loader, val_loader, test_loader

    meta = {
        "P": P, "W": W, "C": C, "L": L,
        "seed": seed,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "X_train_p": X_train_p, "y_train_p": y_train_p,
        "X_val_p": X_val_p, "y_val_p": y_val_p,
        "X_test_p": X_test_p, "y_test_p": y_test_p,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "mu": mu, "sd": sd,
    }

    return train_loader, val_loader, test_loader, meta


#-----------------------------train function-------------------

def train_model_with_loaders(model, train_loader, val_loader,
                             epochs=50, lr=1e-3, patience=7,
                             use_pos_weight=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    pos_weight = None
    if use_pos_weight:
        y_train_all = []
        for _, yb in train_loader:
            y_train_all.append(yb)
        y_train_all = torch.cat(y_train_all).float().to(device)

        pos = float(y_train_all.sum().item())
        neg = float(len(y_train_all) - pos)

        if pos > 0 and neg > 0:
            pos_weight = torch.tensor([neg / pos], device=device)
        else:
            pos_weight = torch.tensor([1.0], device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []
    train_probs = []   # list of tensors — one per epoch
    val_probs   = []   # list of tensors — one per epoch

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        epoch_train_probs = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1).float()

            optimizer.zero_grad()
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits).detach()          # ← collect probs
            epoch_train_probs.append(probs)

            preds = (probs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = loss_sum / total
        train_acc  = correct / total
        train_probs.append(torch.cat(epoch_train_probs))   # shape ≈ [N_train]

        # ---- val ----
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        epoch_val_probs = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).view(-1).float()

                logits = model(xb).view(-1)
                loss = criterion(logits, yb)

                loss_sum += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits).detach()     # ← collect probs
                epoch_val_probs.append(probs)

                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += xb.size(0)

        val_loss = loss_sum / total
        val_acc  = correct / total
        val_probs.append(torch.cat(epoch_val_probs))       # shape ≈ [N_val]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    return (
        model,
        train_losses, val_losses,
        train_accs,   val_accs,
        train_probs,  val_probs,     # ← new
        training_time
    )


# ------------ evaluation function--------------------
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def eval_loader_binary(model, loader, device=None, threshold=0.5, verbose=True, name="LOADER"):
    """
    Evaluate a binary classifier on a given DataLoader.

    Returns:
        y_true: (N,) int array
        y_pred: (N,) int array
        y_prob: (N,) float array  (sigmoid probabilities)
        acc: float (window-level accuracy)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    y_prob_list, y_true_list, y_pred_list = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)

            # logits shape could be (B,) or (B,1) -> make it (B,)
            logits = logits.view(-1)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= threshold).astype(int)

            y_prob_list.append(probs)
            y_pred_list.append(preds)
            y_true_list.append(yb.detach().cpu().numpy().astype(int).reshape(-1))

    y_true = np.concatenate(y_true_list) if len(y_true_list) else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_list) if len(y_pred_list) else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_list) if len(y_prob_list) else np.array([], dtype=float)

    acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")

    if verbose:
        print(f"{name} accuracy (window-level): {acc:.4f}")

    return y_true, y_pred, y_prob, acc

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score)


    # ------------------ One run for multiple experimentones -------------------------
def select_channels(X, full_order, keep_order):
    """
    X: (P,W,C,L), channels in the order of full_order
    keep_order: list of signal names to keep (subset of full_order)
    """
    idx = [full_order.index(s) for s in keep_order]
    return X[:, :, idx, :]
def run_one_experiment(
    X_full, y,
    full_order,
    keep_signals,
    seed,
    Epochs=50,
    lr=1e-3,
    patience=10,
    batch_size=64,
    normalize=True,
    use_pos_weight=True,
    verbose_split=False,
):
    # 1) select channels
    X = select_channels(X_full, full_order, keep_signals)

    # 2) split -> loaders + split_meta
    train_loader, val_loader, test_loader, split_meta = prepare_split_data_loaders(
        X, y,
        test_size=0.2,
        val_size=0.2,
        seed=seed,
        batch_size=batch_size,
        normalize=normalize,
        verbose=verbose_split,
        return_meta=True,
    )

    # 3) model
    model = CNN1D(len(keep_signals))

    # 4) train
    model, trL, vaL, trA, vaA, trP, valP, _ = train_model_with_loaders(
        model,
        train_loader,
        val_loader,
        epochs=Epochs,
        lr=lr,
        patience=patience,
        use_pos_weight=use_pos_weight
    )

    # 5) eval on test loader
    y_te, yhat_te, p_te, _ = eval_loader_binary(model, test_loader, name="TEST", verbose=True)

    test_acc = accuracy_score(y_te, yhat_te)
    test_bal_acc = balanced_accuracy_score(y_te, yhat_te)
    test_f1 = f1_score(y_te, yhat_te, zero_division=0)

    # ROC/AUPRC need both classes present
    if len(np.unique(y_te)) == 2:
        test_roc_auc = roc_auc_score(y_te, p_te)
        test_avg_precision = average_precision_score(y_te, p_te)
    else:
        test_roc_auc = np.nan
        test_avg_precision = np.nan

    return {
        "seed": seed,
        "keep_signals": tuple(keep_signals),
        "n_signals": len(keep_signals),

        "test_acc": float(test_acc),
        "test_bal_acc": float(test_bal_acc),
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_roc_auc),
        "test_avg_precision": float(test_avg_precision),

        "final_train_acc": float(trA[-1]),
        "final_val_acc": float(vaA[-1]),
    }
