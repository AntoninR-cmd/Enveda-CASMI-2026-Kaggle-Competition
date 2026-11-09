# Journal IA — Inès

## Sprint 3 — Évaluation et soumission

### Utilisation 1 — Développement des métriques et du ranking
- **Objectif :** calculer MRR@25 et Recall@k et classer les molécules candidates.
- **Apport de l'IA :** proposition de fonctions Python, tests unitaires et explications du Tanimoto.
- **Vérification humaine :** tests pytest, diagnostic des scores et contrôle du rang de la vraie molécule.
- **Résultat :** ranking fonctionnel et 125 tests réussis.

### Utilisation 2 — Construction de l'évaluation
- **Objectif :** reproduire les conditions du test avec des molécules dev absentes du train.
- **Apport de l'IA :** construction du pipeline KNN, prédiction de fingerprints et agrégation de spectres.
- **Vérification humaine :** comparaison des vraies molécules aux candidats et vérification des métriques.
- **Résultat :** MRR@25 = 0.2423 ; Recall@1 = 0.144 ; Recall@5 = 0.3455 ; Recall@25 = 0.6395.

### Utilisation 3 — Génération de la soumission
- **Objectif :** produire les 25 SMILES classés pour les 400 molécules test.
- **Apport de l'IA :** génération de submission.py et des tests de format.
- **Vérification humaine :** contrôle des dimensions, des identifiants et du nombre de candidats.
- **Résultat :** submission.csv généré avec 400 molécules et 2 colonnes.
- **Limite :** la qualité chimique des candidats et le score Kaggle restent à évaluer.
