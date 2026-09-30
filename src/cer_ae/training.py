from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_mse(prediction, target, mask):
    weights = mask.float()
    return ((prediction - target).pow(2) * weights).sum() / weights.sum().clamp_min(1)


def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    mask_train: np.ndarray | None = None,
    mask_val: np.ndarray | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    patience: int = 5,
    learning_rate: float = 1e-3,
    noise_std: float = 0.0,
    seed: int = 42,
):
    """Boucle explicite, avec early stopping et loss sur points observés."""
    set_seed(seed)
    device = choose_device()
    model = model.to(device)
    train_mask = np.ones_like(x_train, bool) if mask_train is None else mask_train
    val_mask = np.ones_like(x_val, bool) if mask_val is None else mask_val
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(train_mask)),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    mask_val_t = torch.from_numpy(val_mask).to(device)
    history = {"train_loss": [], "val_loss": [], "seconds": 0.0, "device": str(device)}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        losses = []
        for clean, mask in loader:
            clean, mask = clean.to(device), mask.to(device)
            noisy = clean + noise_std * torch.randn_like(clean) if noise_std else clean
            optimizer.zero_grad()
            prediction = model(noisy)
            loss = _masked_mse(prediction, clean, mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val_t)
            val_loss = _masked_mse(val_prediction, x_val_t, mask_val_t).item()
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        if val_loss < best_loss - 1e-7:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    history["seconds"] = time.perf_counter() - started
    history["epochs_run"] = len(history["train_loss"])
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, values: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).float().to(device)
            output = model(batch)
            if isinstance(output, tuple):
                output = output[0]
            result.append(output.cpu().numpy())
    return np.concatenate(result) if result else np.empty_like(values)


def encode(model: nn.Module, values: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).float().to(device)
            latent = model.encode(batch)
            if isinstance(latent, tuple):
                latent = latent[0]
            result.append(latent.cpu().numpy())
    return np.concatenate(result)


def train_conditional_autoencoder(model, x_train, context_train, x_val, context_val, *, epochs=20, batch_size=256, seed=42):
    set_seed(seed); device = choose_device(); model = model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(context_train).float()), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": [], "device": str(device)}
    xv, cv = torch.from_numpy(x_val).float().to(device), torch.from_numpy(context_val).float().to(device)
    for _ in range(epochs):
        model.train(); losses = []
        for x, c in loader:
            x, c = x.to(device), c.to(device); optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x, c), x); loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): history["val_loss"].append(nn.functional.mse_loss(model(xv, cv), xv).item())
        history["train_loss"].append(float(np.mean(losses)))
    history["epochs_run"] = epochs
    return model, history


def predict_conditional(model, values, context):
    device = next(model.parameters()).device; model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values).float().to(device), torch.from_numpy(context).float().to(device)).cpu().numpy()


def train_vae(model, x_train, x_val, *, epochs=20, batch_size=256, beta=1e-3, seed=42):
    set_seed(seed); device = choose_device(); model = model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float()), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": [], "device": str(device)}
    xv = torch.from_numpy(x_val).float().to(device)
    for _ in range(epochs):
        model.train(); losses = []
        for (x,) in loader:
            x = x.to(device); optimizer.zero_grad(); reconstruction, mu, logvar = model(x)
            reconstruction_loss = nn.functional.mse_loss(reconstruction, x)
            kl = -.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + beta * kl; loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            reconstruction, mu, logvar = model(xv)
            val = nn.functional.mse_loss(reconstruction, xv) + beta * (-.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        history["train_loss"].append(float(np.mean(losses))); history["val_loss"].append(float(val))
    history["epochs_run"] = epochs
    return model, history
