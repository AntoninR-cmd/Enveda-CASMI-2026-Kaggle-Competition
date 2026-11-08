# CASMI 2026 — Identification de molécules à partir de spectres MS/MS

Ce projet participe à la compétition Kaggle [Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra). Le but est d'identifier la structure chimique 2D de petites molécules issues d'extraits biologiques à partir de leurs spectres de masse en tandem (MS/MS). Pour chacune des 400 molécules du test, on propose 25 structures candidates (SMILES), classées de la plus probable à la moins probable, et le score est le MRR@25. Comme le test contient des molécules absentes du train, on ne traite pas le problème comme une classification : un modèle apprend à prédire, à partir du spectre, un **fingerprint de Morgan** qui décrit les motifs chimiques de la molécule. Ce fingerprint est ensuite comparé, par similarité de Tanimoto, à celui de structures connues dont la masse correspond à celle déduite du spectre.

## Installation

Prérequis : Python 3.11 ou plus récent (testé avec Python 3.14) et git.

```bash
git clone git@github.com:AntoninR-cmd/projet_ml.git
cd projet_ml
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # version CPU, sans carte NVIDIA
pip install -r requirements.txt
pip install -e .
nbstripout --install             # retire les sorties des notebooks à chaque commit
```

Dans VS Code ou Jupyter, sélectionnez ensuite l'interpréteur / le noyau `.venv`.

> Sans la ligne `pip install torch`, pip téléchargerait sous Linux la version pour carte NVIDIA, de plusieurs Go. Avec une carte NVIDIA, remplacez cette ligne par la commande donnée sur [pytorch.org](https://pytorch.org/get-started/locally/).

## Données

Les données (~2,8 Go) ne sont pas versionnées dans git. Pour les télécharger :

1. Accepter les règles de la compétition sur Kaggle.
2. Enregistrer une clé d'API Kaggle : sur kaggle.com, *Settings → API → Generate New Token*, puis coller le jeton dans `~/.kaggle/access_token` :

   ```bash
   mkdir -p ~/.kaggle
   nano ~/.kaggle/access_token        # coller le jeton, enregistrer
   chmod 600 ~/.kaggle/access_token
   python -c "import kagglehub; print(kagglehub.whoami())"   # doit afficher votre nom d'utilisateur
   ```

   (`kagglehub.login()` ne garde la clé qu'en mémoire : elle est perdue à la fin de la commande.)
3. Télécharger les fichiers dans `data/raw/` :

```bash
python -c "import kagglehub; kagglehub.competition_download('enveda-CASMI26-molecule-id-mass-spectra', output_dir='data/raw')"
```

On obtient `train.parquet`, `test.parquet` et `sample_submission.csv`.

Organisation du dossier `data/` (chemins définis dans `src/casmi/config.py`) :

```
data/
├── raw/         # données Kaggle brutes, jamais modifiées
└── processed/   # fichiers produits par le pipeline (meta.parquet, split.parquet…)
```

## Tests

```bash
pytest
```
