"""Предсказание свойств молекул (MoleculeNet): графовые нейронные сети против
классических моделей на молекулярных отпечатках и дескрипторах.

Запуск: python run_experiment.py ../data ../results [--quick]
"""
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, global_mean_pool
from torch_geometric.utils.smiles import x_map, e_map
from xgboost import XGBClassifier, XGBRegressor

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("NTHREADS", max(1, os.cpu_count() - 1))))

DATASETS = {  # имя PyG -> тип задачи
    "ESOL": "reg",
    "FreeSolv": "reg",
    "Lipo": "reg",
    "BBBP": "cls",
}
SEEDS = [0, 1, 2]
if "--quick" in __import__("sys").argv: SEEDS = [0]
if os.environ.get("SEEDS"): SEEDS = [int(x) for x in os.environ["SEEDS"].split(",")]
GNNS = ["GCN", "GAT", "GIN", "GINE"]
HIDDEN, LAYERS, DROPOUT = 128, 3, 0.1
MAX_EPOCHS, PATIENCE, LR, BATCH = (5, 3, 1e-3, 64) if "--quick" in __import__("sys").argv else (150, 20, 1e-3, 64)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------- разбиения
def scaffold_split(smiles, frac=(0.8, 0.1, 0.1)):
    """Детерминированное scaffold-разбиение (Bemis–Murcko), как в MoleculeNet:
    крупные группы каркасов — в обучение, редкие — в валидацию и тест."""
    groups = {}
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else s
        groups.setdefault(sc, []).append(i)
    sets = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    tr, va, te = [], [], []
    for g in sets:
        if len(tr) + len(g) <= frac[0] * n:
            tr += g
        elif len(va) + len(g) <= frac[1] * n:
            va += g
        else:
            te += g
    return tr, va, te


def random_split(n, seed, frac=(0.8, 0.1, 0.1)):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return list(idx[:a]), list(idx[a:b]), list(idx[b:])


# ---------------------------------------------------------------- признаки для классики
def ecfp(smiles, radius=2, nbits=2048):
    out = np.zeros((len(smiles), nbits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        out[i, list(fp.GetOnBits())] = 1
    return out


def rdkit_desc(smiles):
    names = [d[0] for d in Descriptors._descList]
    out = np.zeros((len(smiles), len(names)), dtype=np.float64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for j, (_, f) in enumerate(Descriptors._descList):
            try:
                out[i, j] = f(m)
            except Exception:  # noqa: BLE001
                out[i, j] = 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -1e6, 1e6)


# ---------------------------------------------------------------- GNN
class CatEncoder(nn.Module):
    """Сумма эмбеддингов категориальных признаков атома/связи (как OGB AtomEncoder)."""

    def __init__(self, cardinalities, dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, dim) for c in cardinalities])

    def forward(self, x):
        return sum(e(x[:, i].clamp(0, e.num_embeddings - 1)) for i, e in enumerate(self.embs))


ATOM_CARD = [len(v) for v in x_map.values()]
BOND_CARD = [len(v) for v in e_map.values()]


class GNN(nn.Module):
    def __init__(self, kind, out_dim):
        super().__init__()
        self.kind = kind
        self.atom = CatEncoder(ATOM_CARD, HIDDEN)
        self.bond = nn.ModuleList([CatEncoder(BOND_CARD, HIDDEN) for _ in range(LAYERS)]) if kind == "GINE" else None
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(LAYERS):
            mlp = lambda: nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))  # noqa: E731
            if kind == "GCN":
                conv = GCNConv(HIDDEN, HIDDEN)
            elif kind == "GAT":
                conv = GATConv(HIDDEN, HIDDEN // 4, heads=4)
            elif kind == "GIN":
                conv = GINConv(mlp(), train_eps=True)
            else:
                conv = GINEConv(mlp(), train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(HIDDEN))
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, out_dim))

    def forward(self, d):
        h = self.atom(d.x)
        for i, (conv, bn) in enumerate(zip(self.convs, self.norms)):
            if self.kind == "GINE":
                h2 = conv(h, d.edge_index, self.bond[i](d.edge_attr))
            else:
                h2 = conv(h, d.edge_index)
            h2 = F.dropout(F.relu(bn(h2)), DROPOUT, self.training)
            h = h + h2  # остаточная связь
        return self.head(global_mean_pool(h, d.batch))


def evaluate_preds(task, y, p):
    if task == "reg":
        return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                    R2=float(r2_score(y, p)))
    return dict(AUC=float(roc_auc_score(y, p)))


def train_gnn(kind, task, ds, tr, va, te, seed, ymean, ystd):
    set_seed(seed)
    model = GNN(kind, 1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tl = DataLoader(ds[tr], batch_size=BATCH, shuffle=True)
    vl = DataLoader(ds[va], batch_size=256)
    tel = DataLoader(ds[te], batch_size=256)

    def predict(loader):
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for b in loader:
                ps.append(model(b).view(-1))
                ys.append(b.y.view(-1))
        p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
        return (p * ystd + ymean if task == "reg" else 1 / (1 + np.exp(-p))), y

    best, best_state, bad, ep_used = None, None, 0, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for b in tl:
            if b.num_graphs < 2:
                continue
            opt.zero_grad()
            out = model(b).view(-1)
            y = b.y.view(-1).float()
            if task == "reg":
                loss = F.mse_loss(out, (y - ymean) / ystd)
            else:
                loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            opt.step()
        p, y = predict(vl)
        score = evaluate_preds(task, y, p)
        val = -score["RMSE"] if task == "reg" else score["AUC"]
        if best is None or val > best:
            best, bad, ep_used = val, 0, ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    p, y = predict(tel)
    res = evaluate_preds(task, y, p)
    res["epochs"] = ep_used
    res["params"] = sum(x.numel() for x in model.parameters())
    return res


def train_classic(name, task, X, y, tr, te, seed):
    if name.startswith("RF"):
        M = RandomForestRegressor if task == "reg" else RandomForestClassifier
        m = M(n_estimators=500, n_jobs=-1, random_state=seed, min_samples_leaf=1)
    else:
        M = XGBRegressor if task == "reg" else XGBClassifier
        m = M(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
              n_jobs=-1, random_state=seed, verbosity=0)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te]) if task == "reg" else m.predict_proba(X[te])[:, 1]
    return evaluate_preds(task, y[te], p)


def main(data_dir, out_dir, quick=False):
    os.makedirs(out_dir, exist_ok=True)
    rows, stats = [], []
    only = os.environ.get("ONLY")
    for name, task in DATASETS.items():
        if only and name not in only.split(","):
            continue
        ds = MoleculeNet(data_dir, name)
        # отбрасываем молекулы без атомов/с некорректным SMILES
        keep = [i for i in range(len(ds)) if ds[i].x is not None and ds[i].x.size(0) > 0 and Chem.MolFromSmiles(ds[i].smiles)]
        ds = ds[keep]
        smiles = [d.smiles for d in ds]
        y = np.array([float(d.y.view(-1)[0]) for d in ds])
        if quick:
            ds, smiles, y = ds[:300], smiles[:300], y[:300]
        print(f"== {name}: {len(ds)} молекул", flush=True)
        X_fp, X_desc = ecfp(smiles), rdkit_desc(smiles)
        sc = scaffold_split(smiles)
        stats.append(dict(dataset=name, task=task, n=len(ds), n_scaffolds=len({MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles}),
                          avg_atoms=float(np.mean([d.x.size(0) for d in ds])), y_mean=float(y.mean()), y_std=float(y.std()),
                          pos_rate=float(y.mean()) if task == "cls" else None))
        for split in ["scaffold", "random"]:
            for seed in SEEDS:
                tr, va, te = sc if split == "scaffold" else random_split(len(ds), seed)
                if task == "cls" and (len(set(y[te])) < 2 or len(set(y[va])) < 2):
                    continue
                ymean, ystd = float(y[tr].mean()), float(y[tr].std() or 1.0)
                for cname, X in [("RF-ECFP", X_fp), ("XGB-ECFP", X_fp), ("RF-Desc", X_desc)]:
                    t0 = time.time()
                    r = train_classic(cname, task, X, y, tr, te, seed)
                    rows.append(dict(dataset=name, split=split, seed=seed, model=cname, time=time.time() - t0, **r))
                for kind in GNNS:
                    t0 = time.time()
                    r = train_gnn(kind, task, ds, tr, va, te, seed, ymean, ystd)
                    rows.append(dict(dataset=name, split=split, seed=seed, model=kind, time=time.time() - t0, **r))
                    print(name, split, seed, kind, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)},
                          f"{time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, "dataset_stats.csv"), index=False)
    json.dump(dict(hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                   lr=LR, batch=BATCH, seeds=SEEDS), open(os.path.join(out_dir, "config.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], quick="--quick" in sys.argv)
