# Partie C — Évaluation, ranking et soumission

## 1. Objectif

La partie C vise à évaluer les performances du système de classement des molécules candidates à partir de spectres MS/MS. L'objectif est de positionner la molécule cible parmi un ensemble de candidats filtrés selon leur masse moléculaire, en utilisant des fingerprints moléculaires et une mesure de similarité.

## 2. Méthodologie

### 2.1. Prédiction des fingerprints par KNN

Le modèle KNN spectral prédit un fingerprint moléculaire à partir des fingerprints associés aux spectres d'entraînement les plus similaires. Les fingerprints sont ensuite agrégés au niveau de la molécule lorsque plusieurs spectres sont disponibles.

### 2.2. Classement des candidats

Les candidats sont d'abord sélectionnés à partir d'un filtrage de masse avec une tolérance exprimée en ppm. Ils sont ensuite classés selon leur similarité de Tanimoto avec le fingerprint prédit.

### 2.3. Métriques d'évaluation

Les performances sont évaluées sur 2 000 molécules du jeu de validation (dev) à l'aide de :
- **MRR@25** : rang moyen réciproque de la molécule cible, limité aux 25 premières positions.
- **Recall@1, Recall@5 et Recall@25** : proportion de molécules retrouvées respectivement dans les 1, 5 et 25 premiers candidats.

## 3. Comparaison avec l'Oracle

L'Oracle utilise le fingerprint exact de la molécule cible pour classer les candidats. Il constitue une borne supérieure théorique des performances du ranking.

Les résultats de l'Oracle montrent que le classement par similarité de fingerprints est très performant lorsque le fingerprint cible est connu. L'écart avec le KNN spectral met en évidence la difficulté de prédire correctement ce fingerprint à partir du spectre MS/MS.

## 4. Optimisation des paramètres KNN

Plusieurs paramètres ont été étudiés : le nombre de voisins, la taille de l'ensemble d'entraînement et la tolérance de masse.

L'augmentation de la taille d'entraînement de 10 000 à 30 000 spectres améliore les performances. Le nombre de voisins influence également le classement : un nombre trop important tend à diluer l'information des spectres les plus proches.

La configuration retenue parmi les paramètres testés est :
- Taille d'entraînement : 30 000 spectres.
- Nombre de voisins : k = 5.
- Tolérance de masse : 5 ppm.

Elle obtient les performances suivantes sur le jeu dev :

| Métrique | Résultat |
|---|---:|
| MRR@25 | 0,3524 |
| Recall@1 | 23,45 % |
| Recall@5 | 49,25 % |
| Recall@25 | 76,95 % |

## 5. Analyse des erreurs

Sur les 2 000 molécules évaluées avec le KNN initial à k=5 et 10 000 spectres :
- 322 molécules sont classées en première position.
- 737 molécules sont présentes dans le top 5.
- 1 973 molécules sont retrouvées dans le top 25.
- 27 molécules sont absentes du top 25.

Ces absences correspondent aux molécules exclues du filtrage par masse. Elles illustrent l'importance du filtrage des candidats dans les performances globales du pipeline.

Les résultats montrent que le système retrouve généralement la molécule cible parmi les candidats, mais que son positionnement en tête de classement reste perfectible.

## 6. Génération de la soumission

Une fonction de génération de soumission a été développée pour appliquer le pipeline aux molécules test. Pour chaque molécule, elle prédit un fingerprint, filtre les candidats par masse, les classe selon leur similarité et conserve jusqu'à 25 SMILES distincts.

Le fichier `submission.csv` a été généré et validé techniquement :
- 400 molécules test.
- 400 identifiants distincts.
- Une ligne par molécule.
- Entre 1 et 25 SMILES par molécule.
- Format CSV avec les colonnes `molecule_id` et `smiles`.

## 7. Limites et perspectives

Les performances restent limitées par la qualité de la prédiction des fingerprints à partir des spectres MS/MS. Des améliorations pourraient être envisagées en explorant d'autres méthodes de prédiction, une meilleure représentation des spectres ou des stratégies d'agrégation plus avancées.

Le filtrage par masse constitue également une étape déterminante : une molécule exclue à ce stade ne peut pas être retrouvée par le ranking.

## 8. Conclusion

Cette partie a permis de construire et d'évaluer un pipeline complet de ranking moléculaire fondé sur les spectres MS/MS, les fingerprints Morgan et la similarité de Tanimoto. L'optimisation des paramètres KNN a amélioré les performances de classement, et une soumission de 400 molécules a été générée et validée techniquement.

Les expérimentations soulignent le potentiel du KNN spectral tout en mettant en évidence les limites de la reconstruction des fingerprints moléculaires.