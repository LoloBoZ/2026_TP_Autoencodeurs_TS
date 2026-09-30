# Auto-encodeurs sur les courbes de charge CER Irlande

Projet pédagogique en français, exécutable sous Windows avec VS Code/Jupyter. Le code compare ACP, auto-encodeurs denses (linéaire et non linéaire), CNN 1D, débruiteur, modèle conditionnel et VAE. Les partitions sont faites par foyer et les métriques de reconstruction privilégient les points initialement observés.

## Données attendues

Les trois sources originales doivent être placées localement dans `data/` :

- `df_join.parquet` : 104 767 778 lignes demi-horaires, colonnes réelles `ID`, `id_pdl`, `time`, `puissance`, `temp` ;
- `df_jour.parquet` : 2 182 663 lignes quotidiennes (`ID`, `jour`, `puissance_moy`, `temp_moy`) ;
- `df_meta_gradient.parquet` : 4 091 foyers et leurs métadonnées.

Les chemins et correspondances sont centralisés dans [`configs/default.yaml`](configs/default.yaml). La grandeur est une puissance et les timestamps sont interprétés en UTC, conformément à la confirmation métier. L'unité exacte de puissance n'étant pas encore précisée, aucune conversion énergétique n'est appliquée. Les textes des métadonnées présentent par ailleurs des caractères de remplacement déjà présents dans le Parquet source.

Le dépôt Git n'inclut **aucune donnée ni aucun résultat calculé** : `data/` et `outputs/` sont ignorés. Il faut disposer séparément des trois Parquet pour reproduire les expériences. Les sorties des notebooks sont aussi retirées avant publication ; les résultats numériques de référence sont résumés dans `RESULTATS.md`.

## Installation

Depuis PowerShell, à la racine du projet :

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

Dans VS Code, sélectionner `.venv\Scripts\python.exe` comme interpréteur et noyau Jupyter. Le projet fonctionne sur CPU et choisit CUDA automatiquement si PyTorch le détecte.

## Ordre de reproduction

```powershell
# 1. Audit hors mémoire des Parquet
.venv\Scripts\python scripts\inspect_data.py --output outputs\reports\data_audit.json

# 2. Tests ciblés
.venv\Scripts\python -m pytest -q

# 3. Pipeline court complet (48 foyers, 2 époques)
.venv\Scripts\python scripts\run_smoke.py --mode smoke

# 4. Prévisualisation de la population full, sans entraînement
.venv\Scripts\python scripts\prepare_selection.py --mode full

# 5. Régénération facultative des notebooks
.venv\Scripts\python scripts\generate_notebooks.py
.venv\Scripts\python scripts\generate_annual_notebook.py

# 6. Ouverture interactive
.venv\Scripts\python -m jupyter lab
```

Pour le cours, commencer par le notebook **`07_courbes_annuelles.ipynb`** : il est autonome et montre la transposition, la normalisation par la moyenne de chaque foyer, le `DataLoader`, l'architecture, la boucle d'entraînement et la comparaison ACP/AE sur une année entière. Les notebooks `00` à `06` restent des compléments thématiques. Les générateurs conservent les sorties quand le code des cellules ne change pas ; le générateur annuel les efface si une cellule de code change, afin d'éviter des résultats périmés.

Avant chaque commit après exécution de notebooks, retirer leurs sorties avec `.venv\Scripts\python scripts\strip_notebook_outputs.py`. Les copies exécutées locales peuvent être conservées sous `outputs/`, ignoré par Git.

## Mode full

```powershell
.venv\Scripts\python scripts\run_smoke.py --mode full
```

Le mode `full` utilise exactement 500 foyers : les 300 plus grandes amplitudes `|gradient|`, puis 200 foyers tirés aléatoirement sans remise parmi les autres avec la graine 42. Le motif de sélection est sauvegardé pour chaque foyer. Le classement peut être changé en `highest` ou `lowest` dans la configuration. Il utilise 60 époques, une période commune du 20/07/2009 au 27/12/2010, les dimensions latentes 2/4/8/16 et l'arrêt anticipé.

L'imputation est désactivée puisque les sources ne contiennent aucune valeur manquante ; une éventuelle fenêtre incomplète serait rejetée plutôt que remplie. L'expérience d'anomalies est également désactivée tant qu'aucune vérité terrain réelle n'est disponible.

La population préparée est consultable dans `data/processed/full_selected_clients.parquet` et son audit dans `outputs/reports/selection_full.json`, avant tout lancement coûteux.

Le mode full a été exécuté avec succès sur une NVIDIA GeForce RTX 3080, via `torch 2.4.1+cu124`. Son rapport est disponible dans `outputs/reports/results_full.json`.

## Structure

- `src/cer_ae/` : configuration, accès Parquet, préparation, modèles, entraînement et évaluation ;
- `notebooks/` : huit supports pédagogiques en français ;
- `tests/` : invariants d'imputation, partitions, fenêtres et dimensions des réseaux ;
- `data/processed/` : matrices, masques, contextes et partitions du smoke test ;
- `outputs/models/` : scalers et poids appris ;
- `outputs/reports/` : audit, configuration et métriques calculées ;
- `outputs/figures/` : figures exportées.

## Garanties et conventions

- Les donneurs J−7/J+7 sont du même foyer et du même créneau, uniquement observés à l'origine, sans récursion.
- Une colonne de partition empêche l'imputation de traverser une frontière temporelle.
- Les fenêtres incomplètes ou au-delà du seuil d'imputation sont rejetées.
- Les statuts observé / deux donneurs / un donneur / toujours manquant sont conservés.
- Les scalers, ACP, encodeurs, clusters et prédicteurs sont ajustés uniquement sur l'apprentissage.
- Aucun foyer n'est commun à l'apprentissage, la validation et le test.
- Le chapitre anomalies est suspendu ; les injections synthétiques ne sont pas utilisées comme validation.

Voir [`RESULTATS.md`](RESULTATS.md) pour la synthèse de l'exécution réalisée.
