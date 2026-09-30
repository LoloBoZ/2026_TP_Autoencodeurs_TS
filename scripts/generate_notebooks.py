"""Génère les sept notebooks pédagogiques à partir du pipeline vérifié."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"; NB.mkdir(exist_ok=True)


def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())


BOOT = r"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks": ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
REPORT = ROOT / "outputs" / "reports" / "results_smoke.json"
if not REPORT.exists():
    raise FileNotFoundError("Exécutez d'abord : .venv\\Scripts\\python scripts\\run_smoke.py --mode smoke")
results = json.loads(REPORT.read_text(encoding="utf-8"))
sns.set_theme(style="whitegrid")
"""


def write(name, title, cells):
    notebook = nbf.v4.new_notebook(cells=[md(f"# {title}\n\nMode utilisé dans ce support : **smoke**, destiné à vérifier la chaîne complète. Les valeurs commentées proviennent du rapport calculé, jamais de scores inventés."), code(BOOT)] + cells)
    notebook.metadata.kernelspec = {"display_name": "Python (.venv)", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    previous_path = NB / name
    if previous_path.exists():
        previous = nbf.read(previous_path, as_version=4)
        executed = {cell.source: cell for cell in previous.cells if cell.cell_type == "code" and cell.get("outputs")}
        for cell in notebook.cells:
            old = executed.get(cell.source)
            if old is not None:
                cell.outputs = old.outputs
                cell.execution_count = old.execution_count
    nbf.write(notebook, NB / name)


write("00_comprendre_preparer.ipynb", "00 — Comprendre et préparer les données", [
md(r"""
## Question étudiée

Comment passer d'un panel demi-horaire massif à des observations comparables, sans confondre absence, changement d'heure et imputation ?

Une **journée** est ici un vecteur de 48 créneaux d'un foyer. Une **semaine** est un vecteur aligné de 336 créneaux. Un **foyer** est une unité statistique qui possède plusieurs journées/semaines : il ne faut donc jamais répartir les journées d'un même foyer entre apprentissage et test.

La variable est une **puissance** et les horodatages sont en **UTC**, conformément à la confirmation métier. L'unité exacte de puissance reste à renseigner : les axes conservent donc l'« unité source ». Multiplier par 0,5 h ne permettra de calculer une énergie que lorsque l'unité aura été confirmée.
"""),
md("""
## Du format long à une matrice d'apprentissage

La source contient une ligne par mesure. Pour un auto-encodeur de journées, on veut **une journée d'un foyer par ligne** et les 48 demi-heures en colonnes. La cellule suivante montre explicitement la transposition sur quelques foyers, puis la création d'un `DataLoader`. Sur des années entières, le même geste produit 17 520 colonnes (voir le notebook 07).
"""),
code(r"""
import torch
from torch.utils.data import TensorDataset, DataLoader
from cer_ae.data import load_half_hourly

ids = pd.read_parquet(ROOT/'data/processed/smoke_selected_clients.parquet', columns=['ID']).ID.head(3).astype(int).tolist()
long = load_half_hourly(ROOT/'data/df_join.parquet', ids, '2009-07-20', '2009-07-23',
                        {'client':'ID', 'timestamp':'time', 'value':'puissance', 'temperature':'temp'})
long['jour'] = long.timestamp.dt.floor('D')
long['creneau'] = long.timestamp.dt.hour*2 + long.timestamp.dt.minute//30
wide = long.pivot(index=['client_id','jour'], columns='creneau', values='value')
assert wide.shape[1] == 48 and wide.notna().all().all()
display(long.head(), wide.iloc[:3,:8])
print('Matrice journées × créneaux :', wide.shape)
loader = DataLoader(TensorDataset(torch.tensor(wide.to_numpy(dtype='float32'))),
                    batch_size=4, shuffle=True)
print('Un mini-lot :', next(iter(loader))[0].shape)
"""),
code(r"""
audit = json.loads((ROOT/'outputs/reports/data_audit.json').read_text(encoding='utf-8'))
pd.DataFrame({k: [v[0] if len(v)==1 else f'{len(v)} lignes'] for k,v in audit.items()}).T
"""),
md(r"""
## Couverture et choix de population

Le diagnostic hors mémoire contrôle schémas, clés, correspondance ID/PDL, plages, grille calendaire et cohérence quotidienne. La sélection par défaut retient les foyers complets sur la période de l'expérience, au lieu de prendre l'intersection aveugle de tous les foyers. Sur le jeu réel, 3 951 des 4 091 foyers couvrent toute la période globale (96,6 %).

La grille observée comporte toujours 48 positions par jour UTC complet. UTC n'ayant pas de transition saisonnière, les changements d'heure locaux irlandais ne créent ici ni journée de 46 ni journée de 50 points.
"""),
code(r"""
windows = np.load(ROOT/'data/processed/smoke_daily_windows.npz')
meta = pd.read_parquet(ROOT/'data/processed/smoke_daily_metadata.parquet')
summary = pd.Series({'foyers': meta.client_id.nunique(), 'journées': len(meta),
                     'points observés': int(windows['observed_mask'].sum()),
                     'points imputés': int((windows['status']>0).sum()),
                     'fenêtres quasi-nulles écartées': results['shape_near_zero_excluded']})
display(summary.to_frame('valeur'))
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(windows['values'][0], marker='.', label='observé')
imputed = windows['status'][0] > 0
ax.scatter(np.flatnonzero(imputed), windows['values'][0][imputed], color='tab:red', label='imputé', zorder=3)
ax.set(xlabel='Créneau demi-horaire', ylabel='Valeur (unité source)', title='Exemple de journée et masque')
ax.legend(); plt.show()
"""),
md(r"""
## Imputation et normalisations

Pour un trou à $t$, la règle rétrospective est

$$\hat x_t = \frac{x_{t-7j}+x_{t+7j}}{2}.$$

Seules les valeurs originellement observées servent de donneurs ; le code conserve quatre états : observé, imputé avec deux donneurs, imputé avec un donneur, toujours manquant. Une colonne de partition peut interdire tout franchissement apprentissage/validation/test. Cette méthode utilise le futur et n'est **pas** temps réel.

Deux vues sont préparées : (1) standardisation commune ajustée sur l'apprentissage, qui préserve les écarts de niveau ; (2) division par la moyenne de la fenêtre, qui isole la forme en conservant le niveau séparément. Les moyennes proches de zéro sont explicitement exclues de la seconde vue.
"""),
code(r"""
from cer_ae.preprocessing import normalize_shape
shape, level, valid = normalize_shape(windows['values'])
fig, axes = plt.subplots(1,2,figsize=(11,3))
axes[0].plot(windows['values'][:8].T, alpha=.5); axes[0].set_title('Niveaux conservés')
axes[1].plot(shape[:8].T, alpha=.5); axes[1].set_title('Formes / moyenne de fenêtre')
for ax in axes: ax.set_xlabel('Créneau'); ax.set_ylabel('Valeur')
plt.show()
"""),
md("""
## Limites et questions aux étudiants

- L'absence de trous dans cette version des données empêche d'évaluer empiriquement l'imputation ; elle est vérifiée par des tests synthétiques ciblés.
- Quelle documentation faudrait-il obtenir avant de convertir la variable `puissance` en kWh ?
- Pourquoi une journée de 46 ou 50 points ne devrait-elle pas être traitée comme une panne de mesure ?
""")
])

write("01_acp_autoencodeur.ipynb", "01 — De l'ACP à l'auto-encodeur", [
md(r"""
## Question étudiée et intuition

Une représentation comprimée peut-elle reconstruire une journée de 48 points ? L'encodeur calcule $z=f_\theta(x)$, le décodeur $\hat x=g_\phi(z)$, et l'apprentissage minimise ici la MSE sur les seuls points initialement observés :

$$\mathcal L(\theta,\phi)=\frac{\sum_{i,t}m_{it}(x_{it}-\hat x_{it})^2}{\sum_{i,t}m_{it}}.$$

Les mini-lots fournissent des gradients bruités mais économiques. La validation et l'arrêt anticipé limitent le surapprentissage. Un auto-encodeur linéaire, sous des hypothèses adaptées, retrouve le **sous-espace** principal de l'ACP ; ses axes latents ne sont pas nécessairement les composantes principales elles-mêmes.
"""),
md("""
## Architecture et boucle d'entraînement explicites

Le pipeline principal entraîne plusieurs modèles et conserve leurs métriques. Ici, on refait une petite expérience en cellules : lecture des journées, séparation par **foyer**, normalisation sur le train, mini-lots, encodeur/décodeur, puis MSE train et validation. Ce code est volontairement lisible ; le notebook 07 reprend le raisonnement à l'échelle annuelle.
"""),
code(r"""
import json, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from cer_ae.preprocessing import normalize_common

raw = np.load(ROOT/'data/processed/smoke_daily_windows.npz')['values']
meta = pd.read_parquet(ROOT/'data/processed/smoke_daily_metadata.parquet')
splits = json.loads((ROOT/'data/processed/smoke_splits.json').read_text(encoding='utf-8'))
indices = {part: np.flatnonzero(meta.client_id.isin(ids)) for part, ids in splits.items()}
assert not (set(splits['train']) & set(splits['validation']))
x, scaler = normalize_common(raw, indices['train'])
train_loader = DataLoader(TensorDataset(torch.tensor(x[indices['train']])), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(x[indices['validation']])), batch_size=128)
print('Journées × 48 :', x.shape, ' ; mini-lot :', next(iter(train_loader))[0].shape)
"""),
code(r"""
class SmallAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 48))

    def forward(self, curves):
        return self.decoder(self.encoder(curves))

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallAE(latent_dim=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = {'train': [], 'validation': []}
for epoch in range(10):
    model.train()
    train_sum, train_n = 0., 0
    for (curves,) in train_loader:
        curves = curves.to(device)
        optimizer.zero_grad()
        loss = ((model(curves) - curves)**2).mean()
        loss.backward()
        optimizer.step()
        train_sum += loss.item() * len(curves)
        train_n += len(curves)
    model.eval()
    val_sum, val_n = 0., 0
    with torch.no_grad():
        for (curves,) in val_loader:
            curves = curves.to(device)
            val_sum += ((model(curves) - curves)**2).mean().item() * len(curves)
            val_n += len(curves)
    history['train'].append(train_sum / train_n)
    history['validation'].append(val_sum / val_n)
fig, ax = plt.subplots(figsize=(7,3))
ax.plot(history['train'], label='train')
ax.plot(history['validation'], label='validation')
ax.set(xlabel='Époque', ylabel='MSE standardisée', title='Entraînement explicite du petit AE')
ax.legend(); plt.show()
"""),
code(r"""
rows=[]
for name, d in results['daily'].items():
    rows.append({'méthode':name, 'RMSE observée':d['rmse_observed'], 'MAE observée':d['mae_observed'],
                 'paramètres':d.get('parameters'), 'secondes':d.get('history',{}).get('seconds')})
comparison=pd.DataFrame(rows).sort_values('RMSE observée')
display(comparison)
ax=comparison.plot.bar(x='méthode', y='RMSE observée', figsize=(10,3), legend=False)
ax.set_ylabel('RMSE standardisée (test)'); plt.xticks(rotation=35,ha='right'); plt.show()
"""),
code(r"""
histories={k:v['history'] for k,v in results['daily'].items() if 'history' in v}
fig,ax=plt.subplots(figsize=(8,3))
for name,h in histories.items(): ax.plot(h['val_loss'],marker='o',label=name)
ax.set(xlabel='Époque',ylabel='MSE validation',title='Courbes d’apprentissage (smoke)'); ax.legend(ncol=2); plt.show()
print('Interprétation calculée :', comparison.iloc[0]['méthode'], 'obtient la plus faible RMSE dans ce smoke test.')
"""),
code(r"""
fig,ax=plt.subplots(figsize=(10,3))
for name in ['pca_2','linear_ae_2','nonlinear_ae_2']:
    ax.plot(np.arange(48)/2,results['daily'][name]['rmse_by_slot'],label=name)
ax.set(xlabel='Heure',ylabel='RMSE standardisée',title='Erreur selon l’heure'); ax.legend(); plt.show()
latent=np.load(ROOT/'data/processed/smoke_daily_latent.npy'); meta=pd.read_parquet(ROOT/'data/processed/smoke_daily_metadata.parquet')
fig,ax=plt.subplots(figsize=(6,4)); sns.scatterplot(x=latent[:,0],y=latent[:,1],hue=meta.season,s=12,alpha=.6,ax=ax); ax.set_title('Espace latent, coloré par saison'); plt.show()
"""),
code(r"""
from IPython.display import Image, display
display(Image(filename=str(ROOT/'outputs/figures/smoke_reconstructions.png')))
"""),
md("""
## Lecture prudente

Deux époques vérifient le pipeline, elles ne suffisent pas à conclure sur la capacité maximale des réseaux. Dans l'exécution observée, l'ACP est devant les auto-encodeurs. C'est compatible avec un signal assez régulier et des réseaux encore sous-entraînés.

Questions : la MSE favorise-t-elle les pics ou les creux ? Comment comparer équitablement les temps de calcul CPU/GPU ? Pourquoi faut-il projeter les latents en 2D plutôt que prendre arbitrairement deux coordonnées d'un latent de dimension 16 ?
""")
])

write("02_clustering.ipynb", "02 — Clustering des journées et des foyers", [
md(r"""
## Question étudiée

Les distances entre profils bruts, coordonnées ACP et latents conduisent-elles aux mêmes groupes ? Nous appliquons le même K-means, avec standardisation ajustée sur l'apprentissage :

$$\min_{C_1,\ldots,C_K}\sum_k\sum_{x_i\in C_k}\|x_i-\mu_k\|_2^2.$$

Les centres sont appris sur l'apprentissage ; validation et test sont affectés aux centres existants. Une silhouette dans deux espaces différents ne suffit pas à déclarer un espace « meilleur ».
"""),
code(r"""
cluster=pd.DataFrame(results['clustering']).T
cluster.index.name='espace_k'; display(cluster)
fig,ax=plt.subplots(figsize=(10,3)); ax.bar(cluster.index,cluster.silhouette_test_same_space)
ax.set(ylabel='Silhouette (espace propre)',title='Diagnostic interne — non comparable seul entre espaces')
plt.xticks(rotation=35,ha='right'); plt.show()
"""),
code(r"""
sizes=pd.DataFrame(cluster.cluster_sizes.tolist(),index=cluster.index).fillna(0)
sizes.plot.bar(stacked=True,figsize=(10,3)); plt.ylabel('Journées test'); plt.title('Effectifs des clusters'); plt.show()
"""),
code(r"""
key='autoencoder_k3'; detail=results['clustering'][key]
fig,ax=plt.subplots(figsize=(10,4))
for k,profile in enumerate(detail['profiles']):
    if profile is None: continue
    med=np.array(profile['median']); q25=np.array(profile['q25']); q75=np.array(profile['q75'])
    ax.plot(med,label=f'cluster {k}'); ax.fill_between(np.arange(48),q25,q75,alpha=.15)
ax.set(xlabel='Créneau',ylabel='Valeur (unité source)',title='Médianes et bandes interquartiles — latent AE, K=3'); ax.legend(); plt.show()
display(pd.Series({k:v['stability_ari_train'] for k,v in results['clustering'].items()},name='ARI entre deux initialisations').to_frame())
display(pd.DataFrame(results['household_groups']['heating_counts']).T.fillna(0).astype(int).rename_axis('groupe foyer'))
"""),
md("""
## Du jour au foyer

Une représentation de foyer s'obtient par ses proportions de journées dans chaque cluster, éventuellement croisées avec saison et type de jour. Cela évite de confondre une journée et un foyer. Les métadonnées servent ensuite à décrire des écarts à la population de référence, pas à baptiser automatiquement un cluster « chauffage électrique » ou « véhicule électrique ».

Limites : ce smoke test ne mesure pas encore la stabilité par bootstrap ; `n_init=10` réduit seulement la sensibilité de K-means à son initialisation. Questions : pourquoi les proportions sont-elles compositionnelles ? Que change le clustering des formes normalisées par rapport aux niveaux ?
""")
])

write("03_cnn_temporalite.ipynb", "03 — Auto-encodeur convolutionnel et temporalité", [
md(r"""
## Question étudiée et notions

Sur des semaines de 336 points, un CNN 1D partage un filtre le long du temps :

$$h_{c,t}=\sigma\!\left(b_c+\sum_{c',j}w_{c,c',j}x_{c',t+j}\right).$$

Les strides compressent la longueur ; trois blocs de stride 2 donnent un champ réceptif croissant. Les convolutions transposées reconstruisent puis le module tronque explicitement à 336 points. Ce partage réduit les paramètres, mais ne rend pas le réseau invariant à l'heure ni insensible aux décalages.
"""),
code(r"""
weekly=pd.DataFrame(results['weekly']).T
weekly['temps_s']=weekly.apply(lambda r:r.get('history',{}).get('seconds') if isinstance(r.get('history'),dict) else np.nan,axis=1)
display(weekly[['rmse_observed','mae_observed','parameters','temps_s']])
weekly.rmse_observed.plot.bar(figsize=(7,3)); plt.ylabel('RMSE test observée'); plt.title('Mêmes semaines et même dimension latente'); plt.show()
"""),
code(r"""
import joblib, torch
from cer_ae.models import ConvAutoencoder1D
from cer_ae.training import predict
w=np.load(ROOT/'data/processed/smoke_weekly_windows.npz')['values']; scaler=joblib.load(ROOT/'outputs/models/weekly_scaler.joblib')
wx=scaler.transform(w.reshape(-1,1)).reshape(w.shape).astype('float32'); dim=min(map(int,[k.split('_')[-1] for k in results['daily'] if k.startswith('pca_')]))
cnn=ConvAutoencoder1D(336,dim); cnn.load_state_dict(torch.load(ROOT/f'outputs/models/weekly_cnn1d_latent{dim}.pt',map_location='cpu',weights_only=True)); rec=predict(cnn,wx[:2])
fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True)
for ax,a,b in zip(axes,wx[:2],rec): ax.plot(a,label='semaine'); ax.plot(b,label='CNN'); [ax.axvline(d*48,color='grey',lw=.5) for d in range(1,7)]; ax.legend()
axes[-1].set_xlabel('Créneau demi-horaire (traits = jours)'); fig.supylabel('Valeur standardisée'); plt.show()
"""),
md("""
## Interprétation et annualité

Dans l'exécution courte, l'ACP est encore la meilleure reconstruction. Le CNN utilise nettement moins de paramètres que le dense hebdomadaire, mais deux époques ne suffisent pas à exploiter son biais inductif.

Une entrée d'une semaine ne peut pas apprendre directement une dynamique annuelle. L'expérience annuelle doit être distincte, sur les séries quotidiennes par foyer et une période commune longue ; elle est préparée par `df_jour`, mais n'a pas été entraînée en mode smoke.

Questions : quel est le champ réceptif exact ? Pourquoi comparer les pics séparément de la RMSE globale ? Comment encoder explicitement l'heure si les décalages sont indésirables ?
""")
])

write("04_prediction_metadonnees.ipynb", "04 — Prédire des métadonnées depuis les représentations", [
md(r"""
## Question étudiée

Le type de chauffage réellement disponible (`CHAUFFAGE`) est-il associé aux profils ? Chaque foyer devient **une seule observation**. Nous comparons la même régression logistique pondérée sur variables métier, ACP, latent, et combinaison :

$$P(Y=k\mid x)=\frac{\exp(\beta_k^\top x)}{\sum_j\exp(\beta_j^\top x)}.$$

Les partitions sont faites par foyer ; encodeur, PCA et normalisation sont ajustés sur les foyers d'apprentissage.
"""),
code(r"""
pred=pd.DataFrame(results['metadata_prediction']).T
display(pred[['balanced_accuracy','macro_f1','majority_balanced_accuracy','n_train_clients','n_test_clients']])
pred[['balanced_accuracy','macro_f1','majority_balanced_accuracy']].plot.bar(figsize=(10,3)); plt.axhline(1/3,color='k',ls='--',lw=1); plt.ylabel('Score test'); plt.show()
"""),
code(r"""
chosen=results['metadata_prediction']['pca']
cm=pd.DataFrame(chosen['confusion_matrix'],index=chosen['classes'],columns=chosen['classes'])
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues'); plt.xlabel('Prédit'); plt.ylabel('Réel'); plt.title('Matrice de confusion — représentation ACP'); plt.show()
display(pd.Series({k:v.get('roc_auc_ovr_macro') for k,v in results['metadata_prediction'].items()},name='ROC-AUC OVR macro'))
"""),
code(r"""
import pyarrow.parquet as pq
meta_source=pd.read_parquet(ROOT/'data/df_meta_gradient.parquet',columns=['ID','CHAUFFAGE'])
display(meta_source.CHAUFFAGE.value_counts(dropna=False).to_frame('effectif complet'))
"""),
md("""
## Interprétation et limites

Le test ne contient que huit foyers en mode smoke. Les scores sous la référence majoritaire sont donc un contrôle de plomberie, pas une conclusion substantielle. Le jeu complet est fortement déséquilibré (`Autre` domine) ; balanced accuracy et macro-F1 sont plus informatives que l'accuracy brute. Une ROC-AUC multiclasses ne sera ajoutée qu'avec assez d'exemples par classe.

Questions : quelles variables métier risquent de fuiter la cible ? Pourquoi une journée étiquetée par son foyer n'est-elle pas une observation indépendante ? Comment fixer un test stable pour une courbe d'apprentissage en nombre de foyers étiquetés ?
""")
])

write("05_debruitage_anomalies.ipynb", "05 — Anomalies : expérience suspendue", [
md(r"""
## Décision expérimentale

Cette expérience est volontairement sortie du périmètre actif. Des pics, plateaux ou décalages injectés synthétiquement peuvent vérifier du code, mais ne constituent pas une validation de détection d'anomalies réelles. Le mode `full` n'entraîne donc ni débruiteur ni détecteur.

La distinction conceptuelle demeure : un débruiteur reçoit $\tilde x$ et vise $x$, tandis qu'un détecteur doit produire un score utile sur des événements réels et rares. Une observation inhabituelle peut être bien reconstruite, et une observation légitime rare peut avoir une forte erreur.
"""),
code(r"""
display(pd.Series(results['anomalies'], name='décision').to_frame())
"""),
md("""
## Conditions de reprise

Avant de réactiver ce chapitre, il faudra une définition métier de l'anomalie, des événements réels annotés ou une stratégie d'évaluation semi-supervisée crédible, puis une calibration des seuils uniquement sur validation. Le code d'injection reste disponible dans `src/cer_ae/evaluation.py` à des fins de tests techniques, sans être utilisé comme preuve de performance.
""")
])

write("06_contexte_vae.ipynb", "06 — Contexte météorologique et bonus VAE", [
md(r"""
## Auto-encodeur conditionnel

Nous calculons $z=f(x)$ puis $\hat x=g(z,c)$ avec $c$ = température moyenne et calendrier sin/cos. La comparaison utilise exactement les mêmes fenêtres. L'architecture ne garantit pas que $z$ soit indépendant de la météo ; changer la température à latent fixé n'est pas une intervention causale.
"""),
code(r"""
baseline=results['daily'][f"nonlinear_ae_{min(map(int,[k.split('_')[-1] for k in results['daily'] if k.startswith('nonlinear_ae_')]))}"]
context=results['context']
display(pd.DataFrame({'sans contexte':baseline,'conditionnel':context}).T[['rmse_observed','mae_observed']])
pd.Series(context['rmse_by_temperature_tercile']).plot.bar(figsize=(8,3)); plt.ylabel('RMSE moyenne'); plt.title('Erreur conditionnelle par tercile de température'); plt.xticks(rotation=20,ha='right'); plt.show()
"""),
md(r"""
## Bonus VAE

Le VAE produit $\mu(x)$ et $\log\sigma^2(x)$, puis

$$z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I),$$

et minimise reconstruction + $\beta D_{KL}(q(z|x)\|\mathcal N(0,I))$. La régularisation facilite interpolations et génération mais peut dégrader la fidélité. Elle ne garantit pas de meilleurs clusters ni de meilleures anomalies.
"""),
code(r"""
pd.DataFrame({'AE':baseline,'VAE':results['vae_bonus']}).T[['rmse_observed','mae_observed']]
"""),
md("""
## Limites et questions

Le smoke test n'a que deux époques et une fenêtre climatique courte ; l'erreur par tranche de température et les épisodes froids doivent être étudiés en mode full. Questions : comment vérifier empiriquement si le latent contient encore la température ? Que signifie interpoler entre deux foyers ? Quel rôle joue $\beta$ dans le compromis netteté/régularité ?
""")
])

print(f"Notebooks écrits dans {NB}")
