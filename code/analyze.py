"""Анализ результатов темы 21: сводные таблицы LaTeX, тесты и рисунки.

Запуск: python analyze.py ../results ../results_lipo ../results_bbbp ../paper ../data
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from torch_geometric.datasets import MoleculeNet  # noqa: E402

MODELS = ["RF-ECFP", "XGB-ECFP", "RF-Desc", "GCN", "GAT", "GIN", "GINE"]
DS = [("ESOL", "reg", "ESOL"), ("FreeSolv", "reg", "FreeSolv"), ("Lipo", "reg", "Lipophilicity"), ("BBBP", "cls", "BBBP")]
COL = {"RF-ECFP": "#b9b8b0", "XGB-ECFP": "#8a8a85", "RF-Desc": "#5c5b55",
       "GCN": "#2a78d6", "GAT": "#1baf7a", "GIN": "#eb6834", "GINE": "#eda100"}


def main(*args):
    *res_dirs, paper, data_dir = args
    os.makedirs(os.path.join(paper, "figures"), exist_ok=True)
    df = pd.concat([pd.read_csv(os.path.join(d, "runs.csv")) for d in res_dirs if os.path.exists(os.path.join(d, "runs.csv"))],
                   ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "split", "seed", "model"], keep="last")
    out = {"n_runs": {f"{a}|{b}": int(v) for (a, b), v in df.groupby(["dataset", "split"]).seed.nunique().items()}}

    # ---- статистика наборов
    rows = []
    for name, task, full in DS:
        ds = MoleculeNet(data_dir, name)
        smi = [d.smiles for d in ds if d.x is not None and d.x.size(0) > 0 and Chem.MolFromSmiles(d.smiles)]
        y = np.array([float(d.y.view(-1)[0]) for d in ds if d.x is not None and d.x.size(0) > 0 and Chem.MolFromSmiles(d.smiles)])
        atoms = [Chem.MolFromSmiles(s).GetNumAtoms() for s in smi]
        sc = {MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smi}
        rows.append(dict(dataset=full, task="классификация" if task == "cls" else "регрессия", n=len(smi),
                         scaffolds=len(sc), atoms=np.mean(atoms),
                         target=f"доля 1: {y.mean():.2f}" if task == "cls" else f"{y.mean():.2f} ± {y.std():.2f}"))
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(paper, "dataset_stats.csv"), index=False)
    lines = [f"{r.dataset} & {r.task} & {r.n} & {r.scaffolds} & {r.atoms:.1f} & {r.target.replace('±', '$\\pm$')} \\\\" for r in st.itertuples()]
    open(os.path.join(paper, "tab_data.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---- главная таблица: метрика (RMSE / AUC), среднее ± std, по разбиениям
    metric = {n: ("RMSE" if t == "reg" else "AUC") for n, t, _ in DS}
    agg = {}
    for split in ["scaffold", "random"]:
        for m in MODELS:
            for name, task, full in DS:
                v = df[(df.dataset == name) & (df.split == split) & (df.model == m)][metric[name]]
                agg[(name, split, m)] = (v.mean() if len(v) else np.nan, v.std() if len(v) > 1 else np.nan)
    lines = []
    for split in ["scaffold", "random"]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{'Scaffold-разбиение' if split == 'scaffold' else 'Случайное разбиение'}}}}} \\\\")
        for m in MODELS:
            row = [m]
            for name, task, full in DS:
                vals = {mm: agg[(name, split, mm)][0] for mm in MODELS if not np.isnan(agg[(name, split, mm)][0])}
                best = (min if task == "reg" else max)(vals, key=vals.get) if vals else None
                mu, sd = agg[(name, split, m)]
                if np.isnan(mu):
                    row.append("--")
                    continue
                s = f"{mu:.3f}"
                if m == best:
                    s = f"\\textbf{{{s}}}"
                if not np.isnan(sd):
                    s += f"\\,{{\\scriptsize$\\pm${sd:.3f}}}"
                row.append(s)
            lines.append(" & ".join(row) + " \\\\")
        if split == "scaffold":
            lines.append("\\midrule")
    open(os.path.join(paper, "tab_main.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    out["agg"] = {f"{k[0]}|{k[1]}|{k[2]}": [None if np.isnan(a) else round(float(a), 4), None if np.isnan(b) else round(float(b), 4)]
                  for k, (a, b) in agg.items()}

    # ---- лучшая GNN vs лучшая классика; разрыв scaffold/random
    summ = {}
    for name, task, full in DS:
        for split in ["scaffold", "random"]:
            ok = lambda ms: {m: agg[(name, split, m)][0] for m in ms if not np.isnan(agg[(name, split, m)][0])}
            g, c, e = ok(["GCN", "GAT", "GIN", "GINE"]), ok(["RF-ECFP", "XGB-ECFP", "RF-Desc"]), ok(["RF-ECFP", "XGB-ECFP"])
            if not g or not c:
                continue
            pick = min if task == "reg" else max
            bg, bc, be = pick(g, key=g.get), pick(c, key=c.get), pick(e, key=e.get)
            summ[f"{name}|{split}"] = dict(best_gnn=bg, best_gnn_val=round(g[bg], 3), best_classic=bc,
                                           best_classic_val=round(c[bc], 3), best_ecfp=be, best_ecfp_val=round(e[be], 3))
    out["best"] = summ
    # разрыв scaffold vs random (средний по моделям, в %)
    gap = {}
    for name, task, full in DS:
        d = []
        for m in MODELS:
            s, r = agg[(name, "scaffold", m)][0], agg[(name, "random", m)][0]
            d.append((s - r) / r * 100 if task == "reg" else (r - s) * 100)
        gap[name] = round(float(np.nanmean(d)), 1)
    out["scaffold_vs_random_gap"] = gap  # для регрессии: рост RMSE в %, для BBBP: падение AUC в п.п.

    # ---- H1: GNN vs ECFP — попарно по всем (набор, разбиение, запуск), знак «лучше»
    def better(a, b, task):
        return a < b if task == "reg" else a > b
    wins = {}
    for g in ["GCN", "GAT", "GIN", "GINE"]:
        for c in ["RF-ECFP", "XGB-ECFP", "RF-Desc"]:
            w = n = 0
            for name, task, full in DS:
                for split in ["scaffold", "random"]:
                    a, b = agg[(name, split, g)][0], agg[(name, split, c)][0]
                    if np.isnan(a) or np.isnan(b):
                        continue
                    n += 1
                    w += better(a, b, task)
            wins[f"{g}>{c}"] = f"{w}/{n}"
    out["wins_cells"] = wins
    # ранги моделей по 8 ячейкам (набор × разбиение)
    rk = []
    for name, task, full in DS:
        for split in ["scaffold", "random"]:
            v = pd.Series({m: agg[(name, split, m)][0] for m in MODELS})
            if v.isna().any():
                continue
            rk.append(v.rank(ascending=(task == "reg")))
    out["mean_rank"] = pd.concat(rk, axis=1).mean(axis=1).round(2).to_dict()
    # H3: GIN/GINE vs GCN на ESOL/FreeSolv по запускам (Уилкоксон по 6 парам)
    h3 = {}
    for a, b in [("GINE", "GCN"), ("GIN", "GCN"), ("GINE", "GIN"), ("GAT", "GCN")]:
        diffs = []
        for name, task, full in DS:
            for split in ["scaffold", "random"]:
                x = df[(df.dataset == name) & (df.split == split) & (df.model == a)].set_index("seed")[metric[name]]
                y = df[(df.dataset == name) & (df.split == split) & (df.model == b)].set_index("seed")[metric[name]]
                j = x.index.intersection(y.index)
                sgn = 1 if task == "reg" else -1
                rel = sgn * (x[j] - y[j]) / y[j] * 100
                diffs += list(rel.values)
        diffs = np.array(diffs)
        h3[f"{a} vs {b}"] = dict(mean_rel_change_pct=round(float(diffs.mean()), 2), better=int((diffs < 0).sum()),
                                 n=len(diffs), p=float(wilcoxon(diffs).pvalue) if len(diffs) > 5 else None)
    out["arch_pairs"] = h3  # отрицательное изменение = лучше (меньше RMSE / больше AUC)

    # ---- рисунок: метрика по моделям и разбиениям
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9))
    for ax, (name, task, full) in zip(axes, DS):
        x = np.arange(len(MODELS))
        for k, split in enumerate(["scaffold", "random"]):
            mu = [agg[(name, split, m)][0] for m in MODELS]
            sd = [0 if np.isnan(agg[(name, split, m)][1]) else agg[(name, split, m)][1] for m in MODELS]
            ax.errorbar(x + (k - 0.5) * 0.3, mu, yerr=sd, fmt="o" if split == "scaffold" else "s", ms=3.5,
                        color="#2a78d6" if split == "scaffold" else "#eb6834", elinewidth=0.8, capsize=1.5,
                        label="scaffold" if split == "scaffold" else "random")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(f"{full}: {metric[name]}{' ↓' if task == 'reg' else ' ↑'}", fontsize=8, loc="left")
        ax.axvline(2.5, color="#bbb", lw=0.6, ls="--")
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)
        ax.tick_params(axis="y", labelsize=7)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(paper, "figures", "metrics.pdf"))
    fig.savefig(os.path.join(res_dirs[0], "metrics.png"), dpi=150)
    plt.close(fig)

    json.dump(out, open(os.path.join(res_dirs[0], "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    df.to_csv(os.path.join(res_dirs[0], "all_runs.csv"), index=False)
    for k, v in out.items():
        if k != "agg":
            print(k, v)


if __name__ == "__main__":
    main(*sys.argv[1:])
