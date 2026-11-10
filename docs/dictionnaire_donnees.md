# Dictionnaire des données

Description de chaque colonne de `train.parquet` (18 colonnes, 2 539 608 spectres) et de `test.parquet` (12 colonnes, 1 213 spectres, 400 molécules).

Les pourcentages de valeurs manquantes (NA) sont calculés **sur tout le fichier**. Le test décrit ici est le test **visible**, un échantillon tiré du train ; il sera remplacé par le test caché lors de l'évaluation Kaggle (voir `fiche_kaggle.md`).

**Légende de la colonne « Feature ? »**
- **oui** : disponible dans le test et utilisable en entrée du modèle ;
- **non** : absente du test, identifiant, ou réponse à prédire ;
- **fuite** : calculée à partir de la vraie structure ; l'utiliser comme feature fausserait la validation ;
- **à risque** : disponible, mais son usage demande une précaution (voir les cas particuliers).

## 1. Vue d'ensemble

| Colonne | Description | Type | Exemple | NA train | NA test | Dans le test ? | Feature ? |
|---|---|---|---|---|---|---|---|
| `ingest_lib` | Bibliothèque spectrale d'origine (11 valeurs) | texte | `enveda-180` | 0 % | — | non | non : absente du test |
| `normalized_smiles` | Structure de la molécule (SMILES normalisé) | texte | `O=C1NC(=O)c2cc(...)cc21` | 0 % | — | non | non : c'est la réponse |
| `inchikey` | InChIKey complète (27 caractères, avec stéréochimie) | texte | `AAALVYBICLMAMA-UHFFFAOYSA-N` | 0 % | — | non | non : c'est la réponse |
| `inchikey14` | 14 premiers caractères de l'InChIKey : identité 2D de la molécule | texte | `AAALVYBICLMAMA` | 0 % | — | non | non : c'est la réponse (clé du split et du score) |
| `molecular_formula` | Formule brute | texte | `C20H15N3O2` | 0 % | — | non | non : déduite de la réponse |
| `ionization_mode` | Mode d'ionisation : `positive` ou `negative` | texte | `positive` | 0 % | 0 % | oui | **oui** |
| `instrument_type` | Type de spectromètre (82 libellés bruts) | texte | `timsTOF` | 1,30 % (+ 0,06 % de faux NA) | 0 % | oui | **oui**, après normalisation |
| `adduct` | Adduct normalisé (ion formé dans la source) | texte | `[M+H]+` | 0 % | 0 % | oui | **oui** (sert aussi à la masse neutre) |
| `adduct_orig` | Adduct tel qu'écrit par la bibliothèque | texte | `[M+H4N]+` | 0 % | — | non | non : absente du test, `adduct` suffit |
| `precursor_mz` | m/z de l'ion précurseur mesuré | nombre | `330.1237` | 0 % | 0 % | oui | **oui** (sert surtout à la masse neutre) |
| `precursor_error_ppm` | Écart entre la masse mesurée et la masse théorique de la vraie structure | nombre | `1.67` | 0,17 % | — | non | **fuite** (utilisable pour nettoyer le train) |
| `ms2_mzs` | m/z des pics du spectre MS/MS | liste de nombres | `[67.041, 73.007, …]` | 0 % | 0 % | oui | **oui** : entrée principale du modèle |
| `ms2_normalized_intensities` | Intensités des pics, normalisées (pic le plus haut = 1) | liste de nombres | `[0.002, 0.000, …]` | 0 % | 0 % | oui | **oui** : entrée principale du modèle |
| `num_peaks` | Nombre de pics du spectre | entier | `52` | 0 % | — | non | **oui**, recalculée par `len(ms2_mzs)` |
| `base_peak_intensity` | Intensité absolue du pic le plus haut, avant normalisation | nombre | `11000701.0` | 45,80 % | 0 % | oui | **à risque** |
| `collision_energy_ev` | Énergie(s) de collision converties en eV | liste de nombres | `[20.0, 40.0, 60.0]` | 13,29 % | 0 % | oui | **oui**, après traitement de la liste |
| `collision_energy_orig` | Énergie de collision telle qu'écrite par la source | texte | `"20,40,60"` | 12,62 % | 0 % | oui | non : texte brut, redondant avec `collision_energy_ev` |
| `collision_energy_orig_units` | Unité d'origine : `eV`, `NCE`, `V` ou `unknown` | texte | `NCE` | 0 % | 0 % | oui | **à risque** |
| `molecule_id` | Identifiant d'une molécule inconnue du test | texte | `m_005e53` | — | 0 % | test seul | non : identifiant (regroupe les spectres d'une requête) |
| `spectrum_id` | Identifiant d'un spectre du test | texte | `s_4d19d522` | — | 0 % | test seul | non : identifiant |

**En résumé :** 10 colonnes sont communes au train et au test. 8 colonnes n'existent que dans le train (la réponse, ses dérivés, et des informations de provenance), et 2 que dans le test (les identifiants). `num_peaks` est la seule colonne absente du test qu'on peut recalculer.

## 2. Cas particuliers

### Identité de la molécule : `inchikey14`, `inchikey`, `normalized_smiles`

- 275 810 `inchikey14` distincts, 275 950 `inchikey` et 277 566 `normalized_smiles`.
- `inchikey14` ne code que la **connectivité 2D**. 139 molécules ont plusieurs `inchikey` (des stéréo-isomères, que le score Kaggle ne distingue pas), et 1 612 ont plusieurs SMILES (écritures différentes d'une même structure 2D).
- C'est donc `inchikey14` qui identifie une molécule : c'est la clé du split (`split.parquet`), du catalogue et du score.

### `adduct` et `adduct_orig`

- Ils diffèrent pour 183 785 spectres (7,2 %). Ce sont les **mêmes ions écrits différemment** : `adduct` est la version normalisée. Exemples : `[M+H4N]+` → `[M+NH4]+` (53 452 spectres), `[M+CH2O2]-` et `[M+CHO2]-` → `[M+CH2O2-H]-` (83 188), `[M-HO]+` → `[M-H2O+H]+` (21 682), `[Cat]+` → `[M]+` (8 770).
- Seul `adduct` existe dans le test : on n'utilise jamais `adduct_orig`.
- 121 adducts dans le train, 7 dans le test visible (`[M+H]+` 79 %, `[M-H]-` 16 %, puis `[M+CH2O2-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M+Cl]-`).

### `instrument_type`

- 82 libellés bruts pour quelques familles d'appareils : `timsTOF` (45 %), `Orbitrap` (29 %), puis `QTOF`, `LC-ESI-QTOF`, `ESI-QFT`, `LC-ESI-QFT`, `LC-ESI-ITFT`, `qTof`… avec des variantes de casse (`Orbitrap` / `orbitrap`, `QTOF` / `qtof` / `qTof`). **À normaliser** en familles avant tout usage.
- **Faux NA** : 1 645 spectres contiennent un texte au lieu d'une vraie valeur manquante (`"nan"` : 972, `"N/A"` : 376, chaîne vide : 297). `isna()` ne les détecte pas : il faut les convertir en NA. Avec eux, 1,36 % des spectres n'ont pas d'instrument.
- Le test visible est **100 % timsTOF**. On ne sait pas si c'est aussi le cas du test caché.

### `precursor_mz` et `precursor_error_ppm`

- `precursor_mz` est la mesure : avec `adduct`, elle donne la masse neutre de la molécule (`compute_neutral_masses`, B1.1). Quelques valeurs aberrantes dans le train : 52 spectres sous 50 et 25 au-dessus de 2 000 (maximum 2,4 millions). Dans le test visible, elle va de 245 à 460.
- `precursor_error_ppm` compare la mesure à la masse théorique **de la vraie structure** : elle n'existe que si l'on connaît la réponse. C'est une **fuite** : jamais comme feature. En revanche, elle mesure la qualité de l'étiquette et peut servir **à nettoyer le train** (A1.4).
- Sa distribution : médiane 1,5 ppm, 92,9 % des spectres sous 10 ppm, mais un 99e centile à 2 884 ppm (maximum 470 millions) : ces valeurs extrêmes signalent des étiquettes fausses (mauvais adduct, mauvaise structure). Sur timsTOF, l'erreur est beaucoup plus faible : 99 % des spectres sous 4,4 ppm.

### Les pics : `ms2_mzs`, `ms2_normalized_intensities`, `num_peaks`

- Deux listes de même longueur par spectre, sans aucun NA ni spectre vide. Elles pèsent 6,3 Go une fois décompressées : on les lit avec `load_spectra(row_ids)`, jamais en entier.
- `num_peaks` vaut `len(ms2_mzs)`. Elle n'est pas dans le test mais se recalcule.
- **Écart important entre train et test** : médiane de 42 pics par spectre dans le train (10 % sous 5 pics), contre 230 dans le test visible (10 % sous 36). À analyser en A1.3.

### `base_peak_intensity`

- Intensité absolue du pic le plus haut avant normalisation. Elle dépend de l'appareil et de la concentration de l'échantillon, pas seulement de la molécule.
- Le NA dépend **entièrement de la bibliothèque** : 0 % de NA pour `enveda-180`, `gnps`, `enveda-np-examples` et `masaryk`, 100 % pour les sept autres. Le simple fait qu'elle manque révèle donc la bibliothèque d'origine.
- Elle est toujours remplie dans le test. **À risque** : un modèle pourrait apprendre « valeur présente = bibliothèque Enveda » plutôt que de la chimie. Stratégies à comparer au sprint 4 (A4.1).

### `collision_energy_ev`, `collision_energy_orig` et `collision_energy_orig_units`

- `collision_energy_ev` est une **liste** : 1 énergie pour 67 % des spectres, plusieurs pour 19,5 % (acquisition « étagée » : 2 à 6 valeurs, par exemple `[20, 40, 60]`), aucune (NA) pour 13,3 %. Les valeurs sont positives et triées, de 0 à 466 eV (médiane 40).
- `collision_energy_orig` est le texte d'origine, parfois signé en mode négatif (`"-60,-40,-20"` dans le test, converti en `[20.0, 40.0, 60.0]`), parfois inexploitable (`"15, 30, 45, 60, 70 or 90 (nominal)"`).
- 17 558 spectres ont un texte d'origine mais pas de valeur en eV, surtout quand l'unité est `NCE` (11 447) : l'énergie normalisée des Orbitrap ne se convertit pas directement en eV.
- `collision_energy_orig_units` : `eV` (54 %), `NCE` (31 %), `unknown` (13 %), `V` (2 %). Toujours `eV` dans le test visible.
- Pour une feature, il faudra résumer la liste (nombre d'énergies, minimum, maximum…) et traiter les NA (voir `meta_features`, A2.2).

## 3. Valeurs manquantes

*Section à compléter en A1.2 : combinaisons de NA, lien avec `ingest_lib` et stratégies proposées pour chaque colonne.*

Colonnes concernées dans le train : `base_peak_intensity` (45,80 %), `collision_energy_ev` (13,29 %), `collision_energy_orig` (12,62 %), `instrument_type` (1,30 %, plus 0,06 % de faux NA) et `precursor_error_ppm` (0,17 %). Aucune colonne n'a de NA dans le test visible.
