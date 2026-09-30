"""Construit le TP annuel autonome, avec code pédagogique visible cellule par cellule."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "07_courbes_annuelles.ipynb"


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md("""
# 07 — Un an de charge par foyer : ACP et auto-encodeur

**Question.** Une courbe de 365 jours à 48 mesures par jour peut-elle être comprimée en 32, 64 ou 128 nombres tout en conservant sa structure annuelle, hebdomadaire et journalière ?

Ici, **une observation est un foyer entier** : 17 520 colonnes demi-horaires. Les résultats des TP précédents portaient sur des jours (48 colonnes) ou des semaines (336 colonnes), et ne répondent donc pas à cette question. Une dimension latente de 128 représente 0,73 % des 17 520 valeurs, mais le réseau lui-même peut contenir plusieurs millions de paramètres : le taux de compression du *code* ne mesure pas la taille du modèle.

Le calcul complet utilise les 500 foyers sélectionnés précédemment. Une carte CUDA accélère beaucoup l'entraînement ; sur CPU, réduire `MAX_CLIENTS` et `EPOCHS` sert seulement à vérifier le déroulement, sans produire une comparaison concluante.
"""),
    code("""
from pathlib import Path
import sys, copy, time
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
MAX_CLIENTS = 500
EPOCHS = 35
PATIENCE = 7
LATENT_DIMS = (32, 64, 128)
START, END = '2010-01-01', '2011-01-01'  # borne finale exclue
N_POINTS = 365 * 48
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}, {N_POINTS=}, {MAX_CLIENTS=}')
"""),
    md("""
## 1. Partir de la table longue

On conserve les foyers de la sélection existante et on lit uniquement l'année 2010 dans le Parquet demi-horaire. Le choix d'une période commune est essentiel : la colonne 100 d'un foyer doit représenter exactement le même instant que la colonne 100 d'un autre. Les horodatages sont en UTC et 2010 possède 365 × 48 créneaux. Les éventuels foyers incomplets seront écartés, sans inventer des valeurs manquantes.
"""),
    code("""
selection = pd.read_parquet(ROOT / 'data/processed/full_selected_clients.parquet', columns=['ID'])
clients = selection['ID'].drop_duplicates().head(MAX_CLIENTS).astype(int).tolist()
if not clients:
    raise ValueError('Sélection vide : exécuter scripts/prepare_selection.py --mode full')

con = duckdb.connect()
con.register('selected_ids', pd.DataFrame({'ID': clients}))
path = (ROOT / 'data/df_join.parquet').as_posix().replace("'", "''")
long = con.execute(f\"\"\"
    SELECT p.ID AS client_id, p.time AS timestamp, p.puissance AS puissance
    FROM read_parquet('{path}') AS p
    SEMI JOIN selected_ids AS s ON p.ID = s.ID
    WHERE p.time >= TIMESTAMP '{START}' AND p.time < TIMESTAMP '{END}'
    ORDER BY p.ID, p.time
\"\"\").fetchdf()
con.close()
print(long.shape)
display(long.head())
"""),
    md(r"""
## 2. Transposer : un foyer par ligne

`pivot` crée la matrice $X \in \mathbb{R}^{n_{foyers}\times17520}$. Les colonnes sont les instants, pas des variables météorologiques. On vérifie la grille et on rejette toute ligne incomplète. Cette étape est visible car c'est elle qui définit l'objet appris par l'auto-encodeur.
"""),
    code("""
long['timestamp'] = pd.to_datetime(long['timestamp'])
if long.duplicated(['client_id', 'timestamp']).any():
    raise ValueError('Plusieurs mesures pour un même foyer et instant')
grid = pd.date_range(START, periods=N_POINTS, freq='30min')
wide = long.pivot(index='client_id', columns='timestamp', values='puissance').reindex(columns=grid)
valid = wide.notna().all(axis=1)
print(f'Foyers complets : {valid.sum()} / {len(wide)} ; incomplets écartés : {(~valid).sum()}')
wide = wide.loc[valid]
if len(wide) < 10:
    raise ValueError('Trop peu de foyers complets pour cette expérience')
X = wide.to_numpy(dtype=np.float32)
assert X.shape[1] == N_POINTS and np.isfinite(X).all()
display(pd.DataFrame({'foyer': wide.index[:3], 'premier_creneau': X[:3, 0],
                      'dernier_creneau': X[:3, -1]}))
print('Matrice foyers × créneaux :', X.shape)
"""),
    md("""
## 3. Découper par foyer, puis normaliser

Il n'y a qu'une ligne par foyer ; les ensembles sont donc indépendants en identité. Pour étudier **la forme seule**, chaque courbe est divisée par **sa propre moyenne annuelle** : $x'_{it}=x_{it}/\\bar x_i$. Une courbe normalisée a une moyenne de 1, tout en conservant ses variations relatives. Nous écartons les moyennes nulles ou quasi nulles, car la division amplifierait démesurément le bruit. Aucune moyenne ou écart-type global n'est ensuite appliqué.

La moyenne de la courbe test est utilisée pour normaliser cette même courbe : c'est légitime pour une tâche de **reconstruction d'une année entièrement observée**. Pour prédire le futur ou combler une période inconnue, il faudrait calculer cette échelle sur la seule partie disponible.

La sélection initiale a été faite sur les gradients de ces foyers. L'évaluation décrit cette population choisie ; elle ne prétend pas représenter sans biais tous les ménages irlandais.
"""),
    code("""
levels = X.mean(axis=1)
valid_level = np.isfinite(levels) & (levels > 1e-6)
print(f'Foyers à moyenne quasi nulle écartés : {(~valid_level).sum()}')
X = X[valid_level]
wide = wide.iloc[np.flatnonzero(valid_level)]
levels = levels[valid_level]
Xs = (X / levels[:, None]).astype(np.float32)
assert np.allclose(Xs.mean(axis=1), 1.0, atol=1e-5)
rows = np.arange(len(Xs))
train_idx, other_idx = train_test_split(rows, test_size=.30, random_state=SEED)
val_idx, test_idx = train_test_split(other_idx, test_size=.50, random_state=SEED)
x_train, x_val, x_test = Xs[train_idx], Xs[val_idx], Xs[test_idx]
print({name: len(indices) for name, indices in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]})
print('Moyennes des courbes normalisées :', np.round(Xs.mean(axis=1)[:5], 5))
print('Moyennes annuelles avant division :', np.round(levels[:5], 4))
"""),
    md("""
## 4. Référence ACP et variance retenue

L'ACP apprend sur les foyers d'apprentissage uniquement. `PCA` de scikit-learn **soustrait la moyenne de chaque colonne temporelle**, calculée sur le train, pour trouver les directions de variance ; il **ne divise pas chaque colonne par son écart-type**. Ce centrage est inhérent à l'ACP : la courbe moyenne est ajoutée lors de la reconstruction. Il n'y a donc pas de deuxième normalisation qui mettrait artificiellement chaque instant à variance 1. L'AE reçoit les mêmes courbes divisées par la moyenne du foyer ; ses biais peuvent apprendre le profil moyen.

Voir la [documentation de `PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) pour la distinction entre centrage et mise à l'échelle.

Ce centrage retire du **calcul des composantes** la courbe moyenne commune à tous les foyers, mais ne l'efface pas des reconstructions. Une saisonnalité identique chez tous les foyers sera surtout dans `pca.mean_`, et non dans leurs coordonnées ACP. Le décodeur AE peut lui aussi apprendre une structure commune sans la coder individuellement dans $z$. L'espace latent décrit principalement les différences **entre foyers**.

La courbe de variance cumulée indique si « 70 composantes pour 95 % » se vérifie **ici**. Avec environ 350 foyers d'apprentissage, l'ACP ne peut apprendre qu'environ 349 directions non nulles. Les latents 32, 64 et 128 sont comparés à rang égal.
"""),
    code("""
n_components = min(128, len(x_train) - 1, X.shape[1])
t0 = time.perf_counter()
pca = PCA(n_components=n_components, svd_solver='randomized', random_state=SEED)
pca.fit(x_train)
print(f'ACP ajustée en {time.perf_counter()-t0:.1f} s')
print('Moyenne par colonne utilisée pour le centrage :', np.round(pca.mean_[:5], 3))
print('Écarts-types des 5 premières colonnes (non ramenés à 1) :',
      np.round(x_train[:, :5].std(axis=0), 3))
cumulative = np.cumsum(pca.explained_variance_ratio_)
threshold = np.flatnonzero(cumulative >= .95)
print('95 % de variance atteint à :', int(threshold[0]+1) if len(threshold) else '> 128 composantes')
print('Variance à 70 composantes :', f'{cumulative[69]:.1%}' if len(cumulative) >= 70 else 'indisponible')
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(np.arange(1, len(cumulative)+1), cumulative)
ax.axhline(.95, color='tab:red', ls='--', label='95 %')
ax.set(xlabel='Composantes ACP', ylabel='Variance expliquée cumulée', ylim=(0, 1.01))
ax.legend(); plt.show()
"""),
    md("""
## 5. Jeu PyTorch et architecture

Le `DataLoader` fournit des mini-lots de courbes annuelles. L'entrée du réseau a la forme `(lot, 1, 17520)`. Quatre convolutions de pas 2 réduisent la longueur d'un facteur 16. Une couche linéaire crée le code $z$ ; le décodeur effectue le chemin inverse. Le réseau reçoit toute l'année à la fois, mais les convolutions locales ont un biais de régularité temporelle. La couche linéaire relie les périodes éloignées de l'année.
"""),
    code("""
train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val)), batch_size=16)
test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=16)
batch = next(iter(train_loader))[0]
print('Forme d’un mini-lot :', tuple(batch.shape))

class AnnualConvAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(8, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.encoded_length = N_POINTS // 16
        flattened = 16 * self.encoded_length
        self.to_latent = nn.Linear(flattened, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder_conv(x.unsqueeze(1))
        return self.to_latent(h.flatten(1))

    def forward(self, x):
        z = self.encode(x)
        h = self.from_latent(z).reshape(-1, 16, self.encoded_length)
        return self.decoder_conv(h).squeeze(1)

example = AnnualConvAE(64)
with torch.no_grad():
    print('Entrée :', tuple(batch.shape), 'code :', tuple(example.encode(batch).shape),
          'sortie :', tuple(example(batch).shape))
print('Paramètres entraînables :', sum(p.numel() for p in example.parameters()))
"""),
    md("""
## 6. Boucle d'entraînement, visible en entier

À chaque mini-lot : calcul de la reconstruction, MSE, rétropropagation, mise à jour Adam. La validation est calculée sans gradient en fin d'époque et sert à conserver les meilleurs poids. Le test reste fermé jusqu'à l'évaluation finale. Ici toutes les courbes admises sont complètes ; aucune perte masquée n'est nécessaire.
"""),
    code("""
def evaluate_loss(model, loader):
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    with torch.no_grad():
        for (curves,) in loader:
            curves = curves.to(device)
            reconstruction = model(curves)
            total_squared_error += torch.sum((reconstruction - curves)**2).item()
            total_values += curves.numel()
    return total_squared_error / total_values

def fit_autoencoder(latent_dim):
    torch.manual_seed(SEED + latent_dim)
    model = AnnualConvAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {'train': [], 'validation': []}
    best_loss, best_weights, stale = float('inf'), None, 0
    for epoch in range(EPOCHS):
        model.train()
        squared_error, n_values = 0.0, 0
        for (curves,) in train_loader:
            curves = curves.to(device)
            optimizer.zero_grad()
            reconstruction = model(curves)
            loss = torch.mean((reconstruction - curves)**2)
            loss.backward()
            optimizer.step()
            squared_error += torch.sum((reconstruction.detach() - curves)**2).item()
            n_values += curves.numel()
        train_loss = squared_error / n_values
        val_loss = evaluate_loss(model, val_loader)
        history['train'].append(train_loss)
        history['validation'].append(val_loss)
        print(f'z={latent_dim:3d} époque {epoch+1:2d} : train {train_loss:.4f}, validation {val_loss:.4f}')
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_weights)
    return model, history

models, histories = {}, {}
for dim in LATENT_DIMS:
    models[dim], histories[dim] = fit_autoencoder(dim)
"""),
    md("""
## 7. Voir les deux courbes de perte

Une perte train qui baisse alors que la validation remonte indique un surapprentissage. Une validation encore décroissante à la dernière époque indique que l'entraînement devrait être prolongé. L'arrêt anticipé sélectionne les poids, pas la dimension latente : pour choisir celle-ci, on regarde les scores de validation avant le test.
"""),
    code("""
fig, axes = plt.subplots(1, len(LATENT_DIMS), figsize=(14, 3), sharey=True)
for ax, dim in zip(axes, LATENT_DIMS):
    ax.plot(histories[dim]['train'], label='train')
    ax.plot(histories[dim]['validation'], label='validation')
    ax.set(title=f'Code {dim}', xlabel='Époque', ylabel='MSE de forme')
    ax.legend()
plt.tight_layout(); plt.show()
"""),
    md("""
## 8. Évaluer une seule fois sur les foyers test

La RMSE est calculée sur les 17 520 points de chaque foyer test, **divisés par la moyenne annuelle du foyer**. Elle mesure donc une erreur de forme, sans unité de puissance. La courbe moyenne train est un contrôle simple. L'ACP et l'AE ont les **mêmes dimensions de code** et voient les mêmes partitions. On rapporte aussi le nombre de paramètres ; une amélioration éventuelle a un coût.
"""),
    code("""
@torch.no_grad()
def reconstruct(model, array):
    model.eval()
    pieces = []
    for start in range(0, len(array), 16):
        curves = torch.from_numpy(array[start:start+16]).to(device)
        pieces.append(model(curves).cpu().numpy())
    return np.concatenate(pieces)

def rmse(reference, estimate):
    return float(np.sqrt(np.mean((reference - estimate)**2)))

baseline = np.broadcast_to(x_train.mean(axis=0), x_test.shape)
rows = [{'méthode': 'Moyenne train', 'dimension': 0, 'RMSE test': rmse(x_test, baseline),
         'paramètres': 0}]
reconstructions = {}
scores = pca.transform(x_test)
for dim in LATENT_DIMS:
    if dim > n_components:
        continue
    pca_reconstruction = scores[:, :dim] @ pca.components_[:dim] + pca.mean_
    ae_reconstruction = reconstruct(models[dim], x_test)
    reconstructions[('ACP', dim)] = pca_reconstruction
    reconstructions[('AE', dim)] = ae_reconstruction
    rows.extend([
        {'méthode': 'ACP', 'dimension': dim, 'RMSE test': rmse(x_test, pca_reconstruction), 'paramètres': 0},
        {'méthode': 'AE convolutionnel', 'dimension': dim, 'RMSE test': rmse(x_test, ae_reconstruction),
         'paramètres': sum(p.numel() for p in models[dim].parameters())},
    ])
comparison = pd.DataFrame(rows)
display(comparison)
"""),
    md("""
## 9. Ce que la RMSE masque : saisonnalités, pics, lissage

On trace le même foyer test sur trois échelles : année (moyennes quotidiennes), semaine et jour. Les courbes restent dans l'échelle **relative à la moyenne annuelle du foyer**. Le **lissage** est mesuré par la variation moyenne $|x_t-x_{t-1}|$ ; un faible score peut signaler la disparition des pics, sans être une qualité en soi. On mesure séparément la RMSE sur les 5 % de points relatifs les plus hauts **de chaque foyer test**. La même observation test sert à toutes les méthodes.
"""),
    code("""
idx = 0  # foyer test fixé avant d'inspecter les reconstructions
curves = {'Mesuré': x_test[idx]}
for dim in LATENT_DIMS:
    curves[f'ACP {dim}'] = reconstructions[('ACP', dim)][idx]
    curves[f'AE {dim}'] = reconstructions[('AE', dim)][idx]

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
for name, curve in curves.items():
    axes[0].plot(curve.reshape(365, 48).mean(axis=1), label=name,
                 lw=1.7 if name == 'Mesuré' else 1, alpha=1 if name == 'Mesuré' else .7)
    axes[1].plot(np.arange(336)/48, curve[8*48:15*48], label=name, alpha=.8)
    axes[2].plot(np.arange(48)/2, curve[8*48:9*48], label=name, alpha=.8)
axes[0].set(xlabel='Jour de 2010', ylabel='Puissance / moyenne annuelle', title='Saisonnalité annuelle')
axes[1].set(xlabel='Jour dans la semaine', ylabel='Puissance / moyenne annuelle', title='Structure hebdomadaire')
axes[2].set(xlabel='Heure UTC', ylabel='Puissance / moyenne annuelle', title='Structure intrajournalière')
axes[0].legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.show()

high = x_test >= np.quantile(x_test, .95, axis=1, keepdims=True)
diagnostics = []
for (method, dim), rec in reconstructions.items():
    diagnostics.append({'méthode': method, 'dimension': dim,
        'variation / variation mesurée': float(np.abs(np.diff(rec, axis=1)).mean() /
                                               np.abs(np.diff(x_test, axis=1)).mean()),
        'RMSE des 5 % hauts': float(np.sqrt(np.mean((rec[high] - x_test[high])**2)))})
display(pd.DataFrame(diagnostics))
"""),
    md("""
## 10. L'espace latent contient-il les saisonnalités ?

Une bonne reconstruction ne garantit pas que les coordonnées latentes correspondent chacune à une saisonnalité identifiable. On peut sonder le code en prédisant, **sur les foyers test**, des caractéristiques calculées sur leurs courbes d'origine : amplitude hiver–été, amplitude semaine–week-end et amplitude moyenne intrajournalière. Un simple modèle linéaire ajusté sur l'apprentissage évalue la facilité de lecture de ces informations. Il ne démontre ni causalité ni disentanglement.
"""),
    code("""
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

@torch.no_grad()
def encode(model, array):
    model.eval()
    return np.concatenate([model.encode(torch.from_numpy(array[i:i+16]).to(device)).cpu().numpy()
                           for i in range(0, len(array), 16)])

calendar = pd.DatetimeIndex(grid)
winter = calendar.month.isin([12, 1, 2])
summer = calendar.month.isin([6, 7, 8])
weekday = calendar.dayofweek < 5
slot = calendar.hour * 2 + calendar.minute // 30

def seasonal_features(array):
    day_profiles = np.stack([array[:, slot == s].mean(axis=1) for s in range(48)], axis=1)
    return np.column_stack([
        array[:, winter].mean(axis=1) - array[:, summer].mean(axis=1),
        array[:, weekday].mean(axis=1) - array[:, ~weekday].mean(axis=1),
        day_profiles.max(axis=1) - day_profiles.min(axis=1),
    ])

feature_names = ['hiver − été', 'semaine − weekend', 'amplitude intrajournalière']
target_train, target_test = seasonal_features(x_train), seasonal_features(x_test)
probe_rows = []
for dim in LATENT_DIMS:
    for method, z_train, z_test in [
        ('ACP', pca.transform(x_train)[:, :dim], scores[:, :dim]),
        ('AE', encode(models[dim], x_train), encode(models[dim], x_test)),
    ]:
        reg = Ridge(alpha=10).fit(z_train, target_train)
        r2 = r2_score(target_test, reg.predict(z_test), multioutput='raw_values')
        probe_rows.append({'méthode': method, 'dimension': dim, **dict(zip(feature_names, r2))})
display(pd.DataFrame(probe_rows))
"""),
    md("""
## Discussion

- À dimension égale, l'AE améliore-t-il la RMSE test, la reconstruction des pics et les trois échelles temporelles ? Une courbe plus lisse peut correspondre à une meilleure généralisation **ou** à une perte d'événements importants.
- Les codes 128/64/32 ne sont pas automatiquement « plus lisses » dans cet ordre : c'est une hypothèse à vérifier sur les courbes et la variation, pas une propriété mathématique universelle.
- Une année par foyer donne environ 350 observations d'apprentissage pour 17 520 variables. Le réseau a beaucoup de paramètres ; répéter la partition, ajouter d'autres années ou d'autres foyers et régler les hyperparamètres sur la validation rendrait la conclusion plus solide.
- Les cycles du calendrier sont présents dans **toutes** les entrées. La sonde teste les différences **entre foyers** de leurs amplitudes saisonnières ; pour isoler plus clairement les composantes temporelles, comparer des modèles explicites à bases de Fourier, STL ou ondelettes serait instructif.
- Pour rechercher un avantage plus net des AE, une tâche de **débruitage ou d'imputation de blocs masqués** sur des courbes de charge peut être plus adaptée que la seule reconstruction fidèle. Il faut masquer uniquement l'entrée et calculer l'erreur sur les valeurs d'origine tenues à l'écart, toujours sur des foyers test.
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python (.venv)", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
if DEST.exists():
    previous = nbf.read(DEST, as_version=4)
    old_code = [cell.source for cell in previous.cells if cell.cell_type == "code"]
    new_code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    if old_code == new_code:
        for cell, old in zip(notebook.cells, previous.cells):
            if cell.cell_type == "code" and old.cell_type == "code":
                cell.outputs = old.outputs
                cell.execution_count = old.execution_count
nbf.validate(notebook)
nbf.write(notebook, DEST)
print(DEST)
