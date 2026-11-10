# Графовые нейронные сети для предсказания свойств молекул (MoleculeNet)

Код и результаты к статье «Графовые нейронные сети для предсказания свойств молекул: сравнение с классическими моделями на данных MoleculeNet» (Ф. С. Кожухаров, ИММКН им. И. И. Воровича ЮФУ).

Сравниваются GCN, GAT, GIN, GINE (PyTorch Geometric) и классические модели — случайный лес и XGBoost на отпечатках ECFP4, случайный лес на дескрипторах RDKit — на наборах ESOL, FreeSolv, Lipophilicity (регрессия, RMSE) и BBBP (классификация, ROC-AUC) при scaffold-разбиении и случайном разбиении 80/10/10.

## Результаты (тест; ESOL и FreeSolv — среднее по 3 запускам, Lipophilicity и BBBP — 1 запуск)

| Модель | ESOL | FreeSolv | Lipophilicity | BBBP (AUC) |
|---|---|---|---|---|
| *Scaffold* | | | | |
| RF-ECFP | 1.640 | 4.349 | 0.876 | 0.707 |
| XGB-ECFP | 1.599 | 4.005 | 0.857 | 0.667 |
| RF-Desc | **0.884** | 2.444 | 0.785 | **0.747** |
| GCN | 1.072 | 2.214 | 0.769 | 0.709 |
| GAT | 1.177 | **1.988** | 0.739 | 0.732 |
| GIN | 1.102 | 2.431 | **0.702** | 0.727 |
| GINE | 1.126 | 2.541 | 0.708 | 0.676 |
| *Random* | | | | |
| RF-ECFP | 1.167 | 1.806 | 0.788 | 0.928 |
| XGB-ECFP | 1.101 | 1.383 | 0.758 | 0.923 |
| RF-Desc | **0.600** | 1.233 | 0.691 | **0.933** |
| GCN | 0.764 | 1.299 | 0.680 | 0.891 |
| GAT | 0.776 | 1.393 | 0.730 | 0.884 |
| GIN | 0.710 | 1.344 | 0.615 | 0.889 |
| GINE | 0.738 | **1.199** | **0.566** | 0.915 |

## Структура

```
code/
  run_experiment.py  загрузка MoleculeNet, разбиения, модели, обучение
  analyze.py         сводные таблицы, сравнения, рисунок
results/             runs.csv (ESOL, FreeSolv), all_runs.csv (все наборы), summary.json, metrics.png
results_lipo/, results_bbbp/   запуски для Lipophilicity и BBBP
```

## Воспроизведение

```bash
pip install torch torch_geometric rdkit xgboost scikit-learn pandas scipy matplotlib
cd code
python run_experiment.py ../data ../results                         # все наборы, 3 запуска
# или по наборам: ONLY=Lipo SEEDS=0 python run_experiment.py ../data ../results_lipo
python analyze.py ../results ../results_lipo ../results_bbbp ../paper ../data
```

Данные MoleculeNet скачиваются автоматически при первом запуске. Обучение выполнялось на CPU без GPU.
