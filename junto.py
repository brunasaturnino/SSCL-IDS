import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from models import SCARF
from datasets import get_dataset, ExampleDataset
from utils import NTXent, store_pandas_df, get_embeddings_labels

# Diretórios
base_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_dir = os.path.join(base_dir, "new_checkpoints")
tmp_dir  = os.path.join(base_dir, "tmp_folder")
log_dir  = os.path.join(base_dir, "new_logs")
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def preprocess(x_train, x_test):
    x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy="median").fit(x_train)
    x_train = pd.DataFrame(imputer.transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)
    scaler = StandardScaler().fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    return x_train, x_test

def train_model(model, criterion, loader, optimizer, device, epochs, writer):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for anchor, pos1, pos2, _ in pbar:
            anchor, pos1, pos2 = anchor.to(device), pos1.to(device), pos2.to(device)
            optimizer.zero_grad()
            emb_a, emb_p = model(anchor, pos1, pos2)
            loss = criterion(emb_a, emb_p)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * anchor.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_loss /= len(loader.dataset)
        writer.add_scalar("Training Loss", epoch_loss, epoch)

def OOD_classifier(train_features, test_features, k=-1, T=0.04):
    train_t = train_features.t()
    batch_size = 500
    cos_mean_list, cos_max_list = [], []
    for i in range(0, test_features.size(0), batch_size):
        batch = test_features[i:i+batch_size]
        sim = torch.mm(batch, train_t)
        cos_max, _ = sim.max(dim=1)
        if k != -1:
            sim, _ = sim.topk(k, dim=1)
        if T != -1:
            sim = (sim - 0.1).div_(T).exp_()
        cos_mean_list.append(sim.mean(dim=1).cpu())
        cos_max_list.append(cos_max.cpu())
    return torch.cat(cos_mean_list), torch.cat(cos_max_list)

def calc_auroc(train_emb, norm_emb, att_emb):
    sin, sin_max = OOD_classifier(train_emb, norm_emb)
    sout, sout_max = OOD_classifier(train_emb, att_emb)
    sin = torch.nan_to_num(sin, nan=0.0)
    sout = torch.nan_to_num(sout, nan=0.0)
    labels = torch.cat([torch.ones_like(sin), torch.zeros_like(sout)])
    print("AUROC (mean):", roc_auc_score(labels.numpy(), torch.cat([sin, sout]).numpy()))
    print("AUROC (max):", roc_auc_score(labels.numpy(), torch.cat([sin_max, sout_max]).numpy()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=45)
    parser.add_argument("--test_size", type=float, default=0.4)
    args = parser.parse_args()

    # Parâmetros fixos do artigo
    corruption_rate = 0.4
    anchor_corruption_rate = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Device:", device)

    x_train, x_test, y_train, y_test = get_dataset(
        args.dataset_path,
        training_with_attacks=False,
        separate_norm_attack=True,
        test_size=args.test_size
    )
    x_train, x_test = preprocess(x_train, x_test)
    store_pandas_df(x_train, os.path.join(tmp_dir, "train_processed.csv"))
    store_pandas_df(x_test, os.path.join(tmp_dir, "test_processed.csv"))

    ds_train = ExampleDataset(x_train.to_numpy(), y_train.to_numpy(), columns=x_train.columns)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SCARF(
        input_dim=ds_train.shape[1],
        emb_dim=args.embedding_dim,
        corruption_rate=corruption_rate,
        anchor_corruption_rate=anchor_corruption_rate,
        mask_rate=0.0,
        anchor_mask_rate=0.0
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = NTXent(temperature=0.5).to(device)

    # Nome do experimento
    log_name = (
        f"scarf1_embdd_dim={args.embedding_dim}_lr={args.lr}_bs={args.batch_size}"
        f"_epochs={args.epochs}_temp=0.5_cr={corruption_rate}_ach_cr={anchor_corruption_rate}"
    )

    writer = SummaryWriter(log_dir + f"/f{log_name}")

    # Arquivo de saída com resultados
    results_path = os.path.join(log_dir, f"{log_name}.txt")
    with open(results_path, "w") as log_file:

        def log_and_print(text):
            print(text)
            log_file.write(text + "\n")

        log_and_print(f"🔧 Device: {device}")
        log_and_print(f"📊 Experimento: {log_name}")
        log_and_print(f"📁 Dataset: {args.dataset_path}")
        log_and_print(f"🧪 Epochs: {args.epochs}, Batch Size: {args.batch_size}")
        log_and_print(f"📎 Ca: {anchor_corruption_rate}, Cp: {corruption_rate}")

        # Treinamento + log
        train_model(model, criterion, dl_train, optimizer, device, args.epochs, writer)
        writer.close()

        # Salva modelo
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "scarf_ssclids.pth"))
        log_and_print("✅ Modelo salvo.")

        # Avaliação
        mask_norm = (y_test == 0).values
        x_norm, x_att = x_test[mask_norm], x_test[~mask_norm]
        y_norm, y_att = y_test[mask_norm], y_test[~mask_norm]

        ds_norm = ExampleDataset(x_norm.to_numpy(), y_norm.to_numpy(), columns=x_norm.columns)
        ds_att  = ExampleDataset(x_att.to_numpy(),  y_att.to_numpy(), columns=x_att.columns)

        dl_norm = DataLoader(ds_norm, batch_size=256)
        dl_att  = DataLoader(ds_att,  batch_size=256)

        emb_tr, _ = get_embeddings_labels(model, dl_train, device, to_numpy=False, normalize=True)
        emb_norm, _ = get_embeddings_labels(model, dl_norm,  device, to_numpy=False, normalize=True)
        emb_att, _  = get_embeddings_labels(model, dl_att,   device, to_numpy=False, normalize=True)
        
        # AUROC evaluation
        sin, sin_max = OOD_classifier(emb_tr, emb_norm)
        sout, sout_max = OOD_classifier(emb_tr, emb_att)
        sin = torch.nan_to_num(sin, nan=0.0)
        sout = torch.nan_to_num(sout, nan=0.0)
        labels = torch.cat([torch.ones_like(sin), torch.zeros_like(sout)])
        auroc_mean = roc_auc_score(labels.numpy(), torch.cat([sin, sout]).numpy())
        auroc_max  = roc_auc_score(labels.numpy(), torch.cat([sin_max, sout_max]).numpy())

        # 🔹 Log no TensorBoard
        writer.add_scalar("AUROC Mean", auroc_mean, global_step=0)
        writer.add_scalar("AUROC Max", auroc_max, global_step=0)

        # 🔹 Log no arquivo .txt
        log_and_print(f"🎯 AUROC (mean): {auroc_mean:.4f}")
        log_and_print(f"🎯 AUROC (max): {auroc_max:.4f}")
        log_and_print(f"📄 Resultados salvos em: {results_path}")

        writer.close()
