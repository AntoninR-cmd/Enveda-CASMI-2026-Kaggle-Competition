# Décisions de l'équipe

Chaque décision indique qui l'a proposée et son statut : *proposée* (à discuter en réunion) ou *validée*.

## Sprint 1

### Catalogue de structures — proposée par B

- Une ligne par `inchikey14`. Si une molécule a plusieurs SMILES, on garde le plus fréquent parmi ses spectres ; à égalité, le premier dans l'ordre alphabétique.
- La formule et la masse exacte sont calculées par RDKit **depuis le SMILES**, et non depuis le précurseur.
- Le catalogue est trié par masse croissante : `cat_idx` suit l'ordre des masses, et les candidats d'une fenêtre de masse forment un intervalle contigu de `cat_idx`.
- Pour une molécule déjà chargée (SMILES avec `[N+]`…), `ExactMolWt` de RDKit retire la masse d'un électron, alors que `compute_neutral_masses` l'inclut. L'écart (environ 1 ppm) est négligeable devant la tolérance et n'est pas corrigé.

Détails : `notebooks/B/02_filtre_masse.ipynb`, section 1.

### Tolérance du filtre par masse : 10 ppm — proposée par B

Sur la validation provisoire, à 10 ppm : rappel du filtre de 95,2 % par spectre et 98,6 % par molécule, avec 41 candidats en médiane (180 au 90e centile). C'est le coude de la courbe « rappel / nombre de candidats » : au-delà, le rappel progresse peu alors que le nombre de candidats double à chaque palier.

À revoir avec l'évaluation de C : comparer le MRR à 10 et à 20 ppm (riken passe de 73,9 % à 87,2 % de rappel à 20 ppm).

Détails : `notebooks/B/02_filtre_masse.ipynb`, section 3.

### Une masse par molécule : la médiane — proposée par B, à discuter avec C

Une molécule a souvent plusieurs spectres, donc plusieurs masses neutres. Prendre la médiane de ces masses fait passer le rappel à 10 ppm de 95,2 % (par spectre) à 98,6 % (par molécule), et protège contre un spectre isolé mal étiqueté.

### Split figé par molécule — proposée par A

- Une ligne par `inchikey14` dans `data/processed/split.parquet` (colonnes `inchikey14`, `split`, `dev`), produit par `python -m casmi.split` et lu avec `load_split()`.
- Règle : les `inchikey14` du train sont triés, puis on tire 10 % des molécules pour `val` avec `np.random.default_rng(42)`, et 2 000 d'entre elles pour `dev`. C'est **exactement la règle provisoire de B** : le split est identique (vérifié), donc tous les chiffres déjà calculés par B restent valables.
- Résultat : `train` 248 229 molécules (2 287 183 spectres), `val` 27 581 molécules (252 425 spectres), dont `dev` 2 000 molécules (18 945 spectres).
- Les deux parties se ressemblent : 9,2 spectres par molécule en moyenne de chaque côté, et les parts des bibliothèques, des adducts et des modes d'ionisation diffèrent de moins de 0,5 point.
- Pas de split par scaffold pour l'instant : d'après B, la validation est déjà exigeante (Tanimoto maximal médian de 0,65 avec le train). L'option reste ouverte au sprint 4.

Détails : `notebooks/A/01_split.ipynb`.

### Valeurs manquantes : imputeurs et stratégies à tester — proposée par A

- Les NA dépendent de la bibliothèque, et le test visible n'en a aucun : l'imputation sert seulement à récupérer des spectres d'entraînement.
- Les imputeurs ont été choisis en masquant des valeurs observées, au hasard et **par bibliothèque entière** (le vrai mécanisme des NA). Ils n'utilisent que des variables présentes dans le test, et ont été appris sur le split `train`.
- `collision_energy_ev` : règle exacte `NCE × precursor_mz / 500` (11 064 NA reconstruits), puis un modèle de gradient boosting (15,3 eV d'erreur moyenne par bibliothèque masquée, contre 16,6 pour une médiane par instrument).
- `instrument_type` : normalisation en 6 familles (`instrument_family`), puis un modèle pour les 1,4 % de NA.
- `base_peak_intensity` : **aucun imputeur fiable**. Sur une bibliothèque jamais vue, l'erreur atteint un facteur 100, pire qu'une médiane. Il faut aussi se méfier de son NA : il révèle la bibliothèque.
- À comparer en A4.1, avec le MRR@25 sur `dev`, puis sur toute la validation et sur timsTOF à part : supprimer ou imputer `base_peak_intensity` ; retirer de l'entraînement ou imputer les spectres sans énergie ou sans instrument.

Détails : `notebooks/A/02_valeurs_manquantes.ipynb` et la section 3 de `docs/dictionnaire_donnees.md`.

### Train vs test : recommandations pour la validation — proposée par A

**Constat.** Les 1 213 spectres du test visible sont des **copies exactes de spectres d'enveda-180** (timsTOF d'Enveda), retrouvés pic par pic. Pour leurs 400 molécules, les organisateurs ont écarté tous les adducts dimères (`[2M+…]`) et gardé environ 2 spectres sur 3 parmi les adducts simples. Le test visible ressemble donc à « enveda-180, adducts simples, environ 3 spectres par molécule ». On suppose que le test caché suit les mêmes règles, sans pouvoir le vérifier.

| | Train complet | enveda-180 | Test visible |
|---|---|---|---|
| timsTOF | 45 % | 100 % | 100 % |
| Adducts | 121 | 14 (29 % de dimères) | 7, tous simples |
| Pics par spectre (médiane) | 42 | 138 | 230 |
| Spectres de moins de 10 pics | 18 % | 4,5 % | 2 % |
| Spectres par molécule (médiane, max) | 6, 1 468 | 6, 46 | 3, 9 |
| Masse des molécules dans [246, 439] Da | 86 % | 99 % | 100 % |

**Recommandations pour C (`build_queries`) :**
1. Garder le split tel quel, mais rapporter deux scores : sur toute la validation, et sur des **requêtes « type test »** (spectres d'enveda-180 avec un adduct du test).
2. Plafonner le nombre de spectres par requête en imitant le test : tirer, avec une graine fixe, un nombre de spectres suivant la distribution du test (1 à 9, médiane 3).
3. Ne jamais évaluer sur le test visible : ses 400 molécules sont dans le train (368 dans notre split `train`, 32 dans `val`), et ses spectres sont des copies de spectres du train.

**Pour le nettoyage (A1.4) et B :**
4. Les spectres de moins de 10 pics et les adducts dimères ne ressemblent pas au test. Ce sont des candidats à écarter de l'entraînement, à tester en A4.1 plutôt qu'à supprimer d'office.
5. Filtre par masse : timsTOF est très précis (99 % des erreurs sous 4,4 ppm). Une tolérance qui dépend de l'instrument est à tester.

Détails : `notebooks/A/03_train_vs_test.ipynb`.

### Nettoyage : `meta.parquet`, `keep` et `query_ok` — proposée par A

`data/processed/meta.parquet` (produit par `python -m casmi.clean`, lu avec `load_clean_meta()`) contient une ligne par spectre du train, avec tous les `row_id`. Ses colonnes sont celles du contrat d'interfaces, plus **deux filtres** :
- **`keep`** : spectre fiable pour l'**entraînement** (aucune règle déclenchée) ;
- **`query_ok`** : spectre utilisable comme **requête de validation**. Seules les règles qui n'utilisent pas la vraie structure comptent, sinon la validation deviendrait plus facile que le test. **Ne jamais filtrer les requêtes avec `keep`.** L'ajout de cette colonne modifie le contrat d'interfaces : à valider en réunion.

| Règle | Ce qu'elle repère | Structure ? | Spectres | En plus des précédentes |
|---|---|---|---|---|
| `adduct_inconnu` | adduct non interprétable (`[Cat]2+`) | non | 15 | 15 |
| `charge_contre_mode` | charge de l'adduct opposée au mode d'ionisation, physiquement impossible | non | 6 087 | 6 087 |
| `erreur_masse` | masse déduite du précurseur à plus de **0,75 Da** de la masse exacte du SMILES | oui | 25 053 | 21 611 |
| `adduct_douteux` | `[M-CH3]-` : précurseurs dispersés, sans formule possible | non | 619 | 619 |
| **Total** | | | | **28 332 (1,12 %)** |

- **Seuil de 0,75 Da** : une erreur d'adduct décale la masse d'au moins un hydrogène (1,008 Da), alors qu'un précurseur arrondi à l'unité (31 % des `precursor_mz` des quadripôles) s'écarte d'au plus 0,5 Da. Les appareils à haute résolution se trompent de quelques ppm (Makarov *et al.*, *Anal. Chem.* 2006 ; Kind & Fiehn, *BMC Bioinformatics* 2006). Le seuil tombe dans le creux de la distribution des écarts. Un seuil de 1 000 ppm donne presque le même résultat (98 % de spectres en commun), mais écarte à tort 550 précurseurs arrondis de petites molécules et laisse passer 85 erreurs sur des molécules de plus de 1 000 Da.
- L'écart est recalculé avec `compute_neutral_masses` et la masse exacte RDKit, et non lu dans `precursor_error_ppm`, décalée d'un électron ou d'un hydrogène pour la plupart des adducts (B1.1).
- **Par bibliothèque** : gnps 4,8 %, riken 4,0 %, drug_plus 3,5 %, mona 2,3 %, massbank 1,6 %, les autres moins de 0,2 %. enveda-180 n'est pas touchée.
- La règle de masse remplace une liste noire pour `[M-H5O3]+` : elle écarte les 214 spectres décalés de −1,418 Da et garde les 16 dont la masse est juste.
- **Hors `keep`** : les spectres de moins de 10 pics et les adducts dimères ont une étiquette juste. Les retirer de l'entraînement est à tester en A4.1 (A1.3).

Détails : `notebooks/A/04_nettoyage.ipynb`.

### Provisoire, en attendant A

- **Chemins** : les données brutes sont lues dans le dossier donné par la variable d'environnement `CASMI_DATA_DIR` (à défaut, `kaggle/input/`), et les fichiers produits sont écrits dans son sous-dossier `processed/` (à défaut, `data/processed/`). À reprendre dans `config.py`.
- **Split de validation** : 10 % des molécules tirées avec `np.random.default_rng(42)` parmi les `inchikey14` triés, et 2 000 d'entre elles pour `dev`. À remplacer par `split.parquet`.

## Sprint 2

### Fingerprints de référence : Morgan, rayon 2, 2048 bits, binaire — proposée par B

- Paramètres `FP_RADIUS = 2` et `FP_SIZE = 2048` dans `src/casmi/fingerprints.py`, à reporter dans `config.py`.
- `catalog_fp.npy` : matrice `(275 810, 2048)` en `uint8`, ligne `i` = `cat_idx` `i`. Elle pèse 565 Mo et reste hors git, dans le même dossier que `catalog.parquet`. `load_catalog_fps(mmap=True)` évite de la charger entièrement en mémoire.
- Une molécule a en moyenne 46 bits à 1 (2,25 %) : il faudra suivre le Tanimoto, la précision et le rappel des bits, pas l'exactitude.

Détails : `notebooks/B/03_analyse_fingerprints.ipynb`.

### Binning des spectres : 0,1 Da jusqu'à 1 000 m/z — proposée par A

`bin_spectrum` (un spectre, vecteur dense) et `bin_spectra` (une liste de spectres, matrice creuse CSR) dans `src/casmi/features.py`. Choix par défaut, tous réglables :

| Choix | Valeur | Raison |
|---|---|---|
| Plage | 0 à `MAX_MZ = 1000` | 98,7 % des pics du train, 100 % du test visible (m/z max 460) |
| Largeur | `BIN_WIDTH = 0.1` Da, soit 10 000 cases | 0,01 Da donnerait 100 000 cases (environ 200 millions de poids pour la première couche d'un MLP) ; 1 Da confondrait des fragments de formules différentes. L'erreur d'un timsTOF (environ 0,002 Da) reste bien plus petite qu'une case |
| Agrégation | maximum | des pics proches dans une case sont en général le même fragment (38 % des pics du train partagent leur case à 0,1 Da) ; la somme gonflerait les amas de bruit |
| Intensités | racine carrée | 81 % des pics du train font moins de 1 % du pic principal ; la racine leur donne plus de poids sans faire remonter le bruit autant qu'un logarithme |
| Normalisation | norme L2 = 1 | le produit scalaire devient la similarité cosinus |

- Pics ignorés : m/z NaN, négatif ou ≥ `MAX_MZ`, intensité NaN ou nulle. 22 spectres sur 50 000 n'ont aucun pic sous 1 000 et donnent un vecteur nul.
- Taille : 96 cases non nulles par spectre en moyenne, soit environ 2 Go en creux pour tout le train, contre 102 Go en dense.
- Option : `neutral_losses` calcule les pertes neutres (`precursor_mz − m/z`), à biner comme un second vecteur.
- À comparer en A4.1 : la largeur (0,01, 0,1 et 1 Da), la transformation (racine, logarithme, aucune) et l'ajout des pertes neutres.

Détails : `notebooks/A/05_binning.ipynb`.

### Features de métadonnées : 31 ou 26 colonnes — proposée par A

`fit_meta_encoder(train, adduct_encoding)` apprend les encodages sur le train seulement, puis `meta_features(df, encoder)` les applique à n'importe quel tableau (train, validation, test). Le résultat est un DataFrame float32 sans NA, dont les noms de colonnes sont les noms des features, toujours dans le même ordre (`src/casmi/features.py`).

| Feature | Encodage | Raison |
|---|---|---|
| `mode_positif` | 1 si mode positif, sinon 0 | un même fragment n'a pas le même sens dans les deux modes |
| `masse_neutre` | masse neutre (`compute_neutral_masses`) / 1 000 | taille de la molécule ; repli sur `precursor_mz` si l'adduct n'est pas interprétable |
| `log_num_peaks` | `log10` du nombre de pics | de 1 à 56 400 pics ; compté depuis `ms2_mzs` dans le test, qui n'a pas de colonne `num_peaks` |
| `ce_min`, `ce_max`, `ce_mean` | énergie de collision en eV / 100 | résumé de la liste ; NA : règle NCE → eV, puis gradient boosting appris sur le train (imputeur retenu en A1.2) |
| `ce_manquante` | 1 si l'énergie est imputée par le modèle | 12,5 % du train (`keep`), 0 % du test visible |
| adduct, `"onehot"` (par défaut) | *one-hot* des 16 adducts qui ont au moins 1 000 spectres, plus `adduct=autre` (17 colonnes) | 99,8 % du train, 100 % du test visible ; un adduct inconnu tombe dans « autre » |
| adduct, `"decompose"` | `adduct_nM`, `adduct_charge`, le nombre de chacun des 8 groupes fréquents, `adduct_groupe=autre`, `adduct_inconnu` (12 colonnes) | les adducts qui partagent un groupe (Na, dimère…) partagent leurs colonnes ; voir ci-dessous |
| `instrument=…` | *one-hot* des 6 familles d'instrument, plus `instrument=Inconnu` | 1,4 % de NA : une catégorie à part plutôt qu'un imputeur qui se trompe souvent |

- Échelles fixes (/ 1 000, / 100, `log10`) : elles ramènent les valeurs autour de 0-1 sans rien apprendre des données.
- Exclues : les fuites (`normalized_smiles`, `inchikey14`, `precursor_error_ppm`), `ingest_lib` (absente du test) et `base_peak_intensity` (proposée exclue, à confirmer). Un test vérifie que ces colonnes ne changent pas le résultat.
- Coût : environ 20 s pour apprendre l'encodeur et 20 s pour encoder les 2,26 millions de spectres `keep` du split `train`.
- À comparer en A4.1 : imputer l'énergie ou retirer les spectres sans énergie de l'entraînement ; les deux encodages de l'adduct (protocole ci-dessous).

**L'adduct décomposé.** `adduct_groups` décrit un adduct par le nombre de molécules M, sa charge signée et le nombre de chaque groupe ajouté (positif) ou perdu (négatif) : `[M-H]-` → nM = 1, charge = -1, H = -1 ; `[2M+Na]+` → nM = 2, charge = +1, Na = 1 ; `[M+CH2O2-H]-` → nM = 1, charge = -1, CH2O2 = 1, H = -1.
- Les groupes sont des blocs chimiques, pas des atomes : le formiate (CH2O2) s'accroche et part en entier.
- Un terme d'un seul élément compte ses atomes (`+Na2` = 2 Na). Les autres sont écrits dans l'ordre de Hill, pour que `NH4` et `H4N` soient le même groupe.
- Groupes gardés (au moins 1 000 spectres du train) : H, Na, CH2O2, H4N, Cl, H2O, K, C2H4O2. Les groupes plus rares sont comptés dans `adduct_groupe=autre` (0,03 % du train).
- L'imputeur d'énergie est le même dans les deux cas : seules les colonnes d'adduct changent (vérifié par un test).

**Protocole de comparaison des deux encodages (A4.1)** :
1. **Une seule différence.** Les deux modèles ont le même split, les mêmes spectres d'entraînement (`train` et `keep`), le même binning, les mêmes autres features, le même modèle, les mêmes hyperparamètres et le même nombre d'époques. Seul `adduct_encoding` change. Chaque encodeur est appris sur le train seulement.
2. **Évaluation sur `dev`** (requêtes `query_ok`), avec le MRR@25 de C. Jamais sur le test visible : ses molécules sont dans le train.
3. **Bruit.** Chaque encodage est entraîné avec 3 graines. L'écart de MRR entre les deux encodages est encadré par un bootstrap apparié sur les molécules de `dev` (1 000 tirages des 2 000 molécules, mêmes molécules pour les deux encodages). On obtient un intervalle à 95 %.
4. **Par sous-groupe**, car on n'attend pas le même effet partout :
   - les adducts simples présents dans le test visible (`[M+H]+`, `[M-H]-`, `[M+CH2O2-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M+Cl]-`). C'est le cas qui compte pour la compétition ;
   - les dimères et les adducts rares, là où la décomposition devrait aider ;
   - les spectres timsTOF.
5. **Confirmation** sur toute la validation, pour l'encodage retenu et l'autre.
6. **Règle de décision proposée.** On garde `decompose` si son gain sur les adducts simples dépasse le bruit, c'est-à-dire si l'intervalle du bootstrap exclut 0 et si le gain est supérieur à l'écart entre graines. Sinon on garde `onehot`, l'encodage par défaut. Si la décomposition n'aide que sur les dimères et les adducts rares, on testera une version mixte : le *one-hot* plus `adduct_nM` et `adduct_charge`.
7. **Pour A2.3.** Les features de métadonnées sont sauvegardées pour les deux encodages, à côté de la matrice des spectres binés, pour que la comparaison n'ait pas à tout recalculer.

Détails : `notebooks/A/06_meta_features.ipynb`.

### Matrices pour les premiers modèles : `dev` et `train_sample` — proposée par A

`python -m casmi.matrices` construit deux sous-ensembles (environ 1 min 30, 5,3 Go de mémoire au maximum) :

| Sous-ensemble | Spectres | Molécules | Filtre | Rôle | Taille sur disque |
|---|---|---|---|---|---|
| `dev` | 18 899 | 2 000 | `query_ok` | évaluer | 8 Mo |
| `train_sample` | 200 000 (tirés avec la graine 42) | 108 705 | split `train` et `keep` | entraîner | 93 Mo |

- **Fichiers, alignés ligne à ligne** dans `data/processed/` :
  - `X_{nom}.npz` : les spectres binés (creux, 10 000 colonnes, norme 1) ;
  - `meta_{nom}_onehot.parquet` et `meta_{nom}_decompose.parquet` : les features de métadonnées des deux encodages de l'adduct ;
  - `rows_{nom}.parquet` : `row_id` et `inchikey14`.
- **Spectres et métadonnées sont séparés**, car le MLP prend les deux alors que la similarité cosinus ne compare que les spectres :
  - `load_matrices(nom)` renvoie `(X, rows, noms des colonnes)` avec X = [spectres | métadonnées] ;
  - `load_matrices(nom, meta=False)` renvoie les spectres seuls ;
  - `load_matrices(nom, adduct_encoding="decompose")` utilise l'autre encodage.
  - La fonction vérifie l'alignement des `row_id`.
- **Les encodeurs de métadonnées** sont appris sur tout le split `train` (`keep`) et sauvegardés. `load_encoder("onehot")` les recharge, pour encoder plus tard toute la validation et le test avec exactement les mêmes catégories.
- **Vérifications** : aucune molécule commune à `dev` et `train_sample` ; un recalcul indépendant donne des lignes identiques. 3 spectres de `dev` et 111 de `train_sample` n'ont aucun pic sous 1 000 m/z : leur vecteur de spectre est nul, mais ils gardent leurs métadonnées.
- **Mémoire** : `train_sample` occupe 156 Mo en mémoire. Tout le split `train` occuperait environ 1,8 Go en creux, contre 90 Go en dense.
- **Changement par rapport au contrat d'interfaces** : le contrat prévoyait `meta_features(df) -> np.ndarray`. La fonction devient `meta_features(df, encoder) -> pd.DataFrame`, car l'encodeur doit être appris sur le train seulement. Les noms des colonnes servent à interpréter les features, et `.to_numpy()` donne le tableau.

Détails : `notebooks/A/07_matrices.ipynb`.

### PyTorch : ajouté à `requirements.txt` — proposée par A

`torch>=2.15` est activé dans `requirements.txt`, car le MLP (`src/casmi/models/mlp.py`) et le futur `make_loader` en ont besoin. Le README indique d'installer d'abord la version CPU, ou la version adaptée pour une carte NVIDIA. Les tests du MLP sont ignorés automatiquement si PyTorch n'est pas installé.

## Points à transmettre

**À A (données et features) :**
- `collision_energy_ev` est une **liste** par spectre : 67 % ont une seule énergie, 19,5 % plusieurs (acquisition « étagée »), 13 % aucune. À prendre en compte dans `meta_features`.
- `instrument_type` a des libellés hétérogènes à normaliser : `timsTOF` (45 % des spectres), `Orbitrap` et `orbitrap`, `QTOF`, `qtof`, `qTof`, `LC-ESI-QTOF`…
- **Valeurs manquantes des métadonnées**, à traiter dans `meta_features` : `base_peak_intensity` manque pour 45,8 % des spectres, `collision_energy_ev` pour 13,3 %, `collision_energy_orig` pour 12,6 % et `instrument_type` pour 1,3 %. Les colonnes `adduct`, `precursor_mz`, `normalized_smiles`, `inchikey14`, `ionization_mode` et les pics sont complètes.
- **Attention aux faux manquants** de `instrument_type` : 1 645 spectres ont un texte au lieu d'une vraie valeur manquante (`"nan"` : 972, `"N/A"` : 376, chaîne vide : 297). `isna()` ne les voit pas : il faut les convertir en vraies valeurs manquantes avant tout traitement.
- Étiquettes douteuses pour le filtre `keep` : voir les conclusions de `notebooks/B/01_masse_neutre.ipynb`.
- Question à trancher : faut-il un split par scaffold en plus du split par molécule ? La validation est déjà exigeante : Tanimoto maximal médian de 0,65 avec le train, et seulement 6,7 % de quasi-jumeaux (≥ 0,9).

**À C (évaluation) :**
- Combiner les masses des spectres d'une molécule par la médiane (voir plus haut).
- Pour l'oracle : dans une fenêtre de 10 ppm, seules 1,5 % des molécules `dev` ont un candidat au fingerprint identique, et le concurrent le plus proche est à 0,34 de Tanimoto en médiane. L'oracle devrait donc être proche du maximum.
