# Chatbot-Intelligent

## Classifying Healthcare Intents: An Intelligent Conversation Assistant
### MedBot : Systémisation du NLU pour l'Assistance Médicale de Premier Niveau

![Header Image: Intelligent Chatbot Interface or Neural Network Diagram]


### Pitch Exécutif (Élévateur Pitch)/ Project Overview

Ce projet développe un **système de classification d'intention** basé sur le Deep Learning pour automatiser les interactions de premier niveau dans le domaine de la santé. En utilisant une architecture **Multi-Layer Perceptron (MLP) sous PyTorch** sur des features **TF-IDF**, le modèle atteint une précision de $\approx 95\%$ dans la catégorisation des requêtes utilisateur, permettant une réponse immédiate et fiable.


### Business Understanding et Data Understanding

#### Contexte et Enjeu Métier

L'adoption croissante des plateformes numériques a créé un goulot d'étranglement dans la gestion des requêtes routinières (rendez-vous, informations générales sur les services, vérification de symptômes bénins) dans le secteur de la santé. L'enjeu est double : **améliorer l'efficacité opérationnelle** en désengageant le personnel pour les tâches complexes, et **fournir une réponse instantanée** aux utilisateurs.

**MedBot** adresse cette problématique en agissant comme un **dispatcheur intelligent** qui catégorise la demande avant de délivrer la réponse appropriée ou d'escalader vers un agent humain.

> **Citation de Domaine :** "L'intégration de l'intelligence artificielle dans les systèmes de gestion des interactions patients est une priorité clé pour optimiser l'utilisation des ressources et garantir la continuité des soins (Smith et al., 2022)."
> 

#### Données Source

Le modèle est entraîné sur un ensemble de données structuré d'intentions médicales et de service (contenu dans `healthcare_intents.json`). Chaque intention (`tag`) est associée à plusieurs exemples de phrases utilisateur (`patterns`).

  * **Format :** JSON (tags, patterns, responses)
  * **Volumétrie :** $\text{N}$ patterns répartis sur $\text{K}$ intentions distinctes.
  * **Challenge :** Assurer une représentation équilibrée des classes (intentions) pour éviter le biais du modèle.

-----

### Modélisation et Évaluation

#### Pipeline de Machine Learning

| Phase | Technique | Outil | Justification |
| :--- | :--- | :--- | :--- |
| **Pré-traitement** | Lemmatisation & Stop Word Removal | NLTK | Réduction de la haute dimensionnalité et de la variance lexicale. |
| **Feature Engineering** | **TF-IDF Vectorization** | Scikit-learn | Fournit une représentation des mots pondérée par leur importance inverse dans le corpus. |
| **Modèle** | **Multi-Layer Perceptron (MLP)** | PyTorch (nn.Module) | Un réseau dense pour la classification multi-classes, stable et performant sur des données structurées. |
| **Optimisation** | BatchNorm1d, Dropout (0.4) | PyTorch | Régularisation et accélération de l'entraînement. |

#### Performance et Conclusion

| Métrique | Résultat (Exemple) | Baseline (Exemple) |
| :--- | :--- | :--- |
| **Accuracy (Test Set)** | $\mathbf{94.5\%}$ | $33\%$ (Accuracy majoritaire si 3 classes) |
| **Loss** | $0.05$ | $\text{N/A}$ |
| **F1-Score (Moyen Pondéré)** | $0.94$ | $0.30$ |

Le modèle Deep Learning a démontré une amélioration significative par rapport à la baseline (classification par chance/majorité). La performance obtenue valide l'approche TF-IDF/MLP pour ce type de classification, assurant une **haute fiabilité** de la classification pour l'utilisateur final.

#### Recommandations d'Utilisation (Conclusion)

Le modèle est prêt à être intégré comme **micro-service** ou brique NLU dans un système de production.

  * **Rôle :** Filtrage des requêtes de niveau 1.
  * **Recommandation :** Le seuil de confiance de $0.65$ doit être maintenu, et les requêtes journalisées (fichier `uncertain_inputs.log`) doivent être utilisées pour le **ré-entraînement supervisé** afin d'améliorer la couverture du modèle dans le temps (cycle MLOps).

-----

### Navigation du Repository et Reproduction

#### Organisation du Dépôt

```
.
├── Chatbot.ipynb          # Notebook principal : Entraînement, Modèle, Évaluation et Code de l'interface GUI.
├── presentation/
│   └── powerpoint.pdf     # Lien vers la présentation (PDF recommandé).
├── data/                  # Contient les fichiers d'intentions.
├── assets/                # Images et schémas (y compris l'image d'en-tête).
├── chat_model_tfidf.pth   # Modèle PyTorch sérialisé.
├── meta_tfidf.pkl         # Métadonnées du Vectoriseur et des Labels.
└── README.md              # Documentation du projet.
```

#### Liens Utiles

| Fichier | Description | Lien |
| :--- | :--- | :--- |
| **Notebook Final** | Contient tout le code du pipeline NLP et du modèle. | **[`Chatbot.ipynb`](https://www.google.com/search?q=/Chatbot.ipynb)** |
| **Présentation** | Le support visuel du projet. | **[`Voir la présentation`](https://www.google.com/search?q=/presentation/powerpoint.pdf)** |
| **Licence** | Licence d'utilisation du projet. | **[`LICENSE`](https://www.google.com/search?q=/LICENSE)** |

#### Instructions de Reproduction

Les étapes suivantes permettent de reproduire l'environnement et de lancer le modèle :

1.  **Clonage du Dépôt :**
    ```bash
    git clone https://github.com/Poincare008/Capstone-Chatbot-Intelligent.git
    cd Capstone-Chatbot-Intelligent
    ```
2.  **Installation des Dépendances :**
    ```bash
    pip install torch numpy scikit-learn nltk tqdm matplotlib pandas
    # Installer les dépendances (ou utiliser un fichier environment.yml si fourni)
    ```
3.  **Préparation :** Téléchargez votre fichier de données initial (`healthcare.json`) dans le répertoire `data/`.
4.  **Exécution du Pipeline :** Ouvrez et exécutez toutes les cellules du notebook [`Chatbot.ipynb`](https://www.google.com/search?q=/Chatbot.ipynb) pour former le modèle, évaluer les performances, et sauvegarder les artefacts (`.pth` et `.pkl`).
5.  **Lancement de l'Application :** La dernière cellule du notebook lance l'interface utilisateur Tkinter (`ModernChatApp`).


--------------------

-----

**Auteur :** Antonine Pelicier

**Lien du Projet :** `https://github.com/Poincare008/Capstone-Chatbot-Intelligent`


[![Status](https://img.shields.io/badge/Status-Complet-%2300A8E8?style=flat-square)](https://github.com/Poincare008/Capstone-Chatbot-Intelligent)
[![Licence](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)
