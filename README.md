<div style="text-align:left;">
    <img src="./img/img.png" alt="DSW LOGO" width="250px" style="margin-left: 0px;"/>
</div>


#### Description du défi 🚴🚴
Ce projet fais office de compétition de machine learning similaire à celles sur Kaggle, au cours duquel les performances des modèles sont stockées sur un tableau de classement.
Les participants travaillent avec deux fichiers : `data_train.csv`, contenant des données d'entraînement, et `data_test.csv`, avec des données pour les prédictions.

#### Description de l'entreprise 📇
[www.datascienceweekly.org](http://www.datascienceweekly.org) cherche à comprendre le comportement des utilisateurs sur leur site Web pour prédire les abonnements à la newsletter. La compétition implique de construire un modèle pour prédire les conversions, en utilisant des données de trafic Web open source. La métrique d'évaluation est le score f1.

#### Objectifs 🎯
- **Partie 1 :** EDA, prétraitement et entraînement du modèle de base.
- **Partie 2 :** Améliorer le score f1 du modèle avec du feature engineering.
- **Partie 3 :** Faire des prédictions sur `data_test.csv` et les soumettre au tableau de classement.
- **Partie 4 :** Analyser les paramètres du meilleur modèle et recommander des améliorations pour augmenter le taux de conversion.

#### Livrable 📬
- Figures d'EDA.
- Modèle entraîné pour la prédiction des conversions.
- Soumission au tableau de classement.
- Analyse des paramètres du meilleur modèle avec des recommandations exploitables.

Dans ce projet, l'accent a été mis sur l'analyse approfondie des `données déséquilibrées` (imbalanced data). Cela a nécessité une démarche méthodique pour garantir des résultats fiables et significatifs. L'objectif principal était de développer des modèles de machine learning capables de prédire efficacement une variable cible mal équilibrée.

La première étape a consisté à identifier le déséquilibre des classes dans les données et à comprendre ses implications sur la performance des modèles. Ensuite, plusieurs approches ont été explorées pour traiter ce déséquilibre, en mettant particulièrement l'accent sur l'utilisation du F1-score comme métrique d'évaluation adaptée.

Concernant les résultats, plusieurs modèles ont affiché des performances prometteuses. Notamment, le modèle `XGBoost` et la `régression logistique` ont été particulièrement solides. De plus, les combinaisons de ces modèles dans un cadre de vote ou de stacking ont entraîné une légère amélioration des performances (`voting_xgb_lr` et `stacking_xgb_lr`). Ces combinaisons de modèles ont permis de capitaliser sur les forces individuelles de chaque algorithme, conduisant ainsi à des performances globales plus élevées.

Un constat important est le `faible taux de conversion` des acheteurs `chinois`, bien qu'ils représentent un groupe démographique significatif. La priorité principale est donc de conseiller à l'équipe produit de réviser la version chinoise du site pour garantir un `contenu adapté`: traductions précises, options de paiement appropriées.

Etant donné que le site réussit à convertir les acheteurs de moins de `40 ans`, l'équipe marketing devrait concentrer ses efforts sur ce groupe à travers des publicités. Il serait bénéfique de représenter le site (retargeting) aux visiteurs qui ont consulté de `nombreuses pages` mais n'ont pas encore franchi le pas, car c'est un indicateur positif de `conversion potentielle`.


En conclusion, ce projet met en lumière l'importance de prendre en compte le déséquilibre des classes lors de la modélisation des données. Les approches et les techniques que nous avons explorées ont permis d'améliorer la performance des modèles et de produire des prédictions plus précises. Cependant, il reste encore des possibilités d'amélioration, notamment en explorant davantage de techniques spécifiques aux données déséquilibrées et en affinant les paramètres des modèles pour obtenir des performances encore meilleures. Des librairies comme `imbalanced-learn` existent et permettent de prendre en charge ces cas d'imbalanced data.
