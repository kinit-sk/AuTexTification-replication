# utils/train_utils.py

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def train_loop(dataloader, model, optimizer, device):
    """Train model for one epoch. Returns F1 score."""
    print("\n[TRAIN] Starting training loop...")

    model.train()
    losses, preds, true_Y = [], [], []

    progress_bar = tqdm(dataloader, ascii=True)
    for batch_idx, XY in enumerate(progress_bar):
        XY = [xy.to(device, non_blocking=True) for xy in XY]
        Xs, Y = XY[:-1], XY[-1]

        raw_pred = model(*Xs)
        pred = model.postprocessing(raw_pred, argmax=True)
        pred_np = pred.astype(int).ravel()

        preds.append(pred_np)
        true_Y.append(Y.cpu().numpy())

        loss = model.compute_loss(raw_pred, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())

        if batch_idx % 50 == 0 and losses:
            avg_loss = float(np.mean(losses))
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    progress_bar.close()

    mean_loss = float(np.mean(losses))
    print(f"[TRAIN] Finished epoch | Mean loss: {mean_loss:.4f}")

    preds = np.concatenate(preds)
    true_Y = np.concatenate(true_Y)

    f1 = f1_score(true_Y, preds, average="macro", zero_division=0)
    print(f"[TRAIN] Epoch F1={f1:.4f}")

    return f1


def eval_loop(dataloader, model, device, test=False):
    """Evaluate model. Returns F1 score."""
    print("\n[EVAL] " + ("Generating predictions..." if test else "Evaluating model..."))

    model.eval()
    preds, true_Y = [], []

    progress_bar = tqdm(dataloader, ascii=True)
    with torch.no_grad():
        for XY in progress_bar:
            XY = [xy.to(device, non_blocking=True) for xy in XY]
            Xs, Y = XY[:-1], XY[-1]

            output = model(*Xs)
            pred = model.postprocessing(output, argmax=True)
            pred_np = pred.astype(int).ravel()

            preds.append(pred_np)
            true_Y.append(Y.cpu().numpy())

    progress_bar.close()

    preds = np.concatenate(preds)
    true_Y = np.concatenate(true_Y)

    f1 = f1_score(true_Y, preds, average="macro", zero_division=0)
    print(f"[EVAL] F1 score: {f1:.4f}")

    return f1
