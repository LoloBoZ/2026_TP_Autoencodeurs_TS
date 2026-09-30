# Synthèse des résultats réellement obtenus

## Audit des données

L'audit complet des Parquet a été exécuté. Les 4 091 identifiants `ID` et `id_pdl` concordent, et les trois tables couvrent les mêmes foyers. Aucune clé client–horodatage n'est dupliquée, conflictuelle ou manquante. La période va du 15/07/2009 au 01/01/2011 23:30.

Il existe 2 182 662 journées de 48 points et une seule journée terminale de 2 points. Les agrégats quotidiens correspondent aux moyennes recalculées depuis le demi-horaire à environ 10⁻¹⁹ près. 3 951 foyers (96,6 %) couvrent toute la période ; les autres ont surtout une fin anticipée. Cette structure justifie de sélectionner les foyers complets par expérience plutôt que de raccourcir tout le panel.

La variable est confirmée comme une puissance et les horodatages comme UTC. L'unité exacte de puissance reste à préciser ; les calculs demeurent donc en unité source. Les jours sont des jours UTC de 48 créneaux, sans transition DST.

## Smoke test exécuté

Environnement : CPU, graine 42, 48 foyers stratifiés par type de chauffage, période de huit semaines, 2 époques. Les partitions contiennent 33 foyers d'apprentissage, 7 de validation et 8 de test. Le pipeline a préparé 2 688 journées et 384 semaines ; aucun point n'était manquant ou imputé, et aucune fenêtre n'était quasi nulle.

Sur les journées standardisées, les RMSE test observées sont :

| Méthode | latent 2 | latent 4 |
|---|---:|---:|
| ACP | 0,586 | 0,529 |
| AE linéaire | 0,719 | 0,745 |
| AE non linéaire | 0,721 | 0,718 |

Sur les semaines, à dimension latente 2 : ACP 0,628, AE dense 0,717, CNN 1D 0,721. L'ACP est donc meilleure dans ce smoke test. Deux époques vérifient la chaîne, mais ne constituent pas une comparaison convergée.

Le clustering K-means a été évalué dans chaque espace avec affectation du test aux centres appris. Les silhouettes, profils médians, bandes interquartiles, représentants réels, proportions saisonnières, tailles et ARI entre deux initialisations sont enregistrés. Ces silhouettes restent des diagnostics internes à chaque espace, pas une preuve de supériorité entre espaces.

Pour `CHAUFFAGE`, le test de huit foyers donne une balanced accuracy de 0,611 pour la représentation ACP, contre 0,111 pour les variables métier, 0,111 pour le latent AE et 0,222 pour la combinaison. La référence majoritaire vaut 0,333. Ce très petit test est trop instable pour une conclusion métier.

L'expérience d'anomalies a été retirée du périmètre actif : les injections synthétiques ne sont pas considérées comme une validation suffisante. Le mode full ne l'exécutera pas.

Le modèle conditionnel obtient une RMSE 0,710 et le VAE 0,703, contre 0,721 pour l'AE non conditionnel de dimension 2. Ces écarts après deux époques ne démontrent ni indépendance météorologique, ni causalité, ni supériorité du VAE.

## Limites et suites

- Exécuter le mode full avec arrêt anticipé avant toute conclusion comparative.
- Confirmer seulement l'unité exacte de puissance ; le fuseau UTC est désormais renseigné.
- Évaluer l'imputation sur des trous masqués artificiellement, puisque le fichier fourni est complet.
- Répéter la validation annuelle sur d'autres partitions de foyers ; une semaine seule ne porte pas la dynamique annuelle.
- Répéter les partitions par foyer et fournir des intervalles d'incertitude, notamment pour la cible chauffage déséquilibrée.
- Valider les anomalies sur des événements réels annotés ; les injections synthétiques ne suffisent pas.

## Exécution full — RTX 3080

Le mode full a été exécuté sur CUDA avec `torch 2.4.1+cu124` et une NVIDIA GeForce RTX 3080. La population comprend 500 foyers : 300 aux plus grandes amplitudes `|gradient|` et 200 tirés aléatoirement parmi les autres. Les partitions par foyer donnent 350 foyers d'apprentissage, 75 de validation et 75 de test. Le pipeline contient 262 500 journées et 37 500 semaines.

### Reconstruction des journées

| Dimension latente | ACP | AE linéaire | AE non linéaire |
|---:|---:|---:|---:|
| 2 | 0,718 | 0,741 | **0,696** |
| 4 | 0,656 | 0,660 | **0,621** |
| 8 | 0,551 | 0,553 | **0,535** |
| 16 | **0,413** | 0,413 | 0,413 |

Les valeurs sont les RMSE standardisées sur les points observés du test. L'AE non linéaire améliore l'ACP aux faibles dimensions, mais l'écart disparaît à dimension 16. Les AE linéaires convergent vers des erreurs très proches de l'ACP, conformément à la relation attendue entre leurs sous-espaces reconstruits.

### Semaines et modèles complémentaires

À dimension latente 2, les RMSE hebdomadaires sont 0,785 pour l'ACP, 0,779 pour l'AE dense et 0,783 pour le CNN 1D. Le CNN utilise 11 611 paramètres contre 87 122 pour le dense, sans avantage net de reconstruction dans cette expérience.

Le modèle conditionnel obtient une RMSE de 0,690 contre 0,696 pour l'AE non conditionnel latent 2. Le VAE obtient 0,692. Ces différences sont descriptives et ne démontrent ni effet causal de la température ni supériorité générale du VAE.

### Clustering et métadonnées

La meilleure silhouette échantillonnée est celle du latent AE avec K=3 : 0,486 sur 5 000 journées test, avec une stabilité ARI de 0,998 entre deux initialisations. Cette silhouette reste propre à son espace et ne suffit pas à choisir seule une partition.

Pour la cible `CHAUFFAGE`, les balanced accuracies sur 75 foyers test sont 0,361 pour les variables métier, 0,276 pour l'ACP, 0,328 pour le latent AE et **0,455** pour la combinaison. La référence majoritaire vaut 0,333. La combinaison apporte donc le meilleur résultat observé, mais une répétition des partitions reste nécessaire pour quantifier l'incertitude.

L'expérience d'anomalies est restée désactivée comme convenu.

## Expérience annuelle — 2010, 17 520 points par foyer

Le notebook `07_courbes_annuelles.ipynb` a été exécuté sur la sélection full. Parmi les 500 foyers, 499 ont les 17 520 demi-heures complètes de 2010 ; la partition par foyer comprend 349 foyers train, 75 validation et 75 test. Chaque courbe est divisée par **sa propre moyenne annuelle** afin d'étudier la forme. Aucun foyer n'a dû être écarté pour moyenne quasi nulle. L'ACP et les AE convolutionnels reçoivent les mêmes courbes et sont évalués sur les mêmes foyers test. L'ACP centre les colonnes temporelles par leur moyenne d'apprentissage mais ne les divise pas par leur écart-type.

| Dimension du code | ACP : RMSE test | AE convolutionnel : RMSE test |
|---:|---:|---:|
| 32 | **1,084** | 1,094 |
| 64 | **1,069** | 1,085 |
| 128 | **1,055** | 1,083 |

Les RMSE ci-dessus sont sans unité, dans l'échelle « puissance / moyenne annuelle du foyer ». La courbe moyenne du train donne 1,217. Dans cette configuration, l'ACP est meilleure aux trois dimensions. Avec 70 composantes, elle explique 52,1 % de la variance d'apprentissage ; le seuil de 95 % n'est pas atteint avec 128 composantes. Ce résultat dépend de cette population, de la granularité et de la normalisation, et ne contredit pas un seuil de 70 composantes sur un autre panel.

Le rapport entre variation demi-horaire reconstruite et variation réelle passe de 0,383 à 0,474 pour l'ACP de 32 à 128 dimensions, et de 0,225 à 0,278 pour l'AE. Les AE sont donc plus lisses dans ce test. Sur les 5 % de points relatifs les plus hauts de chaque foyer, les RMSE valent 3,649 / 3,561 / 3,598 pour l'AE contre 3,675 / 3,603 / 3,532 pour l'ACP : le diagnostic des pics est mixte. La taille du code n'entraîne pas une amélioration monotone de la RMSE AE.

Une sonde linéaire des amplitudes saisonnières lues dans le latent obtient des résultats mixtes : les codes AE 32/64/128 rendent l'écart semaine–week-end plus lisible que l'ACP de même taille, tandis que l'ACP 64/128 décrit un peu mieux l'écart hiver–été. Ces sondes et la visualisation des trois échelles sont dans le notebook. Une seule partition et une seule architecture AE ne suffisent pas à conclure sur l'ensemble des auto-encodeurs. L'apport pédagogique de cette expérience est aussi de constater qu'une réduction non linéaire peut lisser les événements et perdre face à une référence linéaire solide.
