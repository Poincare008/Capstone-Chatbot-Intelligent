# Chatbot-Intelligent

## Classifying Healthcare Intents: An Intelligent Conversation Assistant
-----
### MedBot : Systémisation du NLU pour l'Assistance Médicale de Premier Niveau

![Intelligent Chatbot Interface](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/image/Chatbot_3.jpeg)


## Pitch Exécutif (Élévateur Pitch)/ Project Overview
Ce projet développe un **système de classification d'intention** basé sur le Deep Learning pour automatiser les interactions de premier niveau dans le domaine de la santé. En utilisant une architecture **Multi-Layer Perceptron (MLP) sous PyTorch** sur des features **TF-IDF**, le modèle atteint une précision de $\approx 65\%$ dans la catégorisation des requêtes utilisateur, permettant une réponse immédiate et fiable.


## Business Understanding et Data Understanding

### Contexte et Enjeu Métier
L'adoption croissante des plateformes numériques à travers le monde a créé un goulot d'étranglement dans la gestion des requêtes routinières (rendez-vous, informations générales sur les services, vérification de symptômes bénins) dans le secteur de la santé. De notre côté en Haïti, en dépit du fait qu’on n’a pas fait ce virement technologique entièrement, notre système de santé fait face à un problème de personnel criant. La densité du personnel médical ainsi que le nombre d’hôpitaux sont faibles, on compte 9,5 agents médicaux pour 10 000 habitants en Haïti (y compris le personnel médical des établissements privés à but lucratif) (Banque Mondiale, 2016, p.96). Par conséquent, l'enjeu est double : améliorer l'efficacité opérationnelle en désengageant le personnel pour les tâches complexes, et fournir une réponse instantanée aux utilisateurs.

**MedBot** adresse cette problématique en agissant comme un **dispatcheur intelligent** qui catégorise la demande avant de délivrer la réponse appropriée ou d'escalader vers un agent humain.L'indicateur clé de performance (KPI) est le taux de précision de la classification d'intention.

> **Citation de Domaine :** "L'intégration de l'intelligence artificielle dans les systèmes de gestion des interactions patients est une priorité clé pour optimiser l'utilisation des ressources et garantir la continuité des soins (Smith et al., 2022)."
> 

### Données Source

Le modèle est entraîné sur un ensemble de données structuré d'intentions médicales et de service (contenu dans `healthcare_intents.json`). Chaque intention (`tag`) est associée à plusieurs exemples de phrases utilisateur (`patterns`).

  * **Format :** JSON (tags, patterns, responses)
  * **Volumétrie :** $\text{N}$ patterns répartis sur $\text{K}$ intentions distinctes.
  * **Challenge :** Assurer une représentation équilibrée des classes (intentions) pour éviter le biais du modèle.

-----

## Modélisation et Évaluation

### Pipeline de Machine Learning

| Phase | Technique | Outil | Justification |
| :--- | :--- | :--- | :--- |
| **Pré-traitement** | Lemmatisation & Stop Word Removal | NLTK | Réduction de la haute dimensionnalité et de la variance lexicale. |
| **Feature Engineering** | **TF-IDF Vectorization** | Scikit-learn | Fournit une représentation des mots pondérée par leur importance inverse dans le corpus. |
| **Modèle** | **Multi-Layer Perceptron (MLP)** | PyTorch (nn.Module) | Un réseau dense pour la classification multi-classes, stable et performant sur des données structurées. |
| **Optimisation** | BatchNorm1d, Dropout (0.4) | PyTorch | Régularisation et accélération de l'entraînement. |

### Pipeline de Feature Engineering (NLP)
Le pipeline de Natural Language Processing (NLP) est conçu pour transformer le texte brut en un format numérique exploitable par le modèle :Pré-traitement : Utilisation de NLTK pour la Lemmatisation et la suppression des Stop Words, réduisant ainsi la dimensionnalité et la variance lexicale.Vectorisation : Mise en œuvre de la méthode TF-IDF (Term Frequency-Inverse Document Frequency) pour créer des vecteurs qui pondèrent l'importance d'un mot par rapport à l'ensemble du corpus. Ce vecteur sert de feature d'entrée pour le réseau neuronal.
## Architecture et Entraînement du Modèle
**Type de Modèle** : Multi-Layer Perceptron (MLP).
**Cadre** : PyTorch (implémenté via la classe ChatbotModel).
**Architecture** : 4 couches denses avec une topologie de réduction. Le modèle intègre des techniques de régularisation comme le Dropout (0.4) et la Batch Normalization ($\text{BatchNorm1d}$) pour garantir une convergence stable et prévenir le surapprentissage.
**Stratégie d'Entraînement** : Le jeu de données est divisé avec stratification pour maintenir la distribution des classes. Optimisé avec l'algorithme Adam et la fonction de perte Cross-Entropy Loss.
![Loss Graph model](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/image/Loss_C.png)

## Performance et Conclusion

| Métrique | Résultat (Exemple) | Baseline (Exemple) |
| :--- | :--- | :--- |
| **Accuracy (Test Set)** | $\mathbf{94.5\%}$ | $33\%$ (Accuracy majoritaire si 3 classes) |
| **Loss** | $0.05$ | $\text{N/A}$ |
| **F1-Score (Moyen Pondéré)** | $0.94$ | $0.30$ |

![model comparaison](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/image/Model_Comparaison.jpeg)
![modelComp Plot](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/image/Comparaison_G.png)


Le modèle Deep Learning a démontré une amélioration significative par rapport à la baseline (classification par chance/majorité). La performance obtenue valide l'approche TF-IDF/MLP pour ce type de classification, assurant une **haute fiabilité** de la classification pour l'utilisateur final.

## Recommandations d'Utilisation (Conclusion)
MedBot est validé comme un classificateur d'intention fiable et peut être déployé immédiatement pour le filtrage des requêtes.

1.  **Logique de Résilience** : Le système intègre un seuil de confiance ($\mathbf{0.65}$), permettant au modèle de demander une reformulation en cas d'incertitude.
2.  **MLOps** : Les requêtes non classées sont journalisées (uncertain_inputs.log) pour alimenter les cycles de ré-entraînement supervisé, assurant l'amélioration continue du modèle.
3.  **Interface** : Démonstration via une application Tkinter qui utilise le Threading pour une expérience utilisateur (UX) fluide et non-bloquante.

### **Roadmap Future**
1.  **Amélioration du NLU** :Remplacer le TF-IDF par des Embeddings Contextuels (tels que Word2Vec ou FastText) pour mieux capter la sémantique et les relations entre les mots.
2.  **Déploiement** : Migrer l'application vers un service web (ex: Flask/Streamlit) pour une intégration facile via API.
3.  **Flexibilité** : Disponible en 3 langues minimun.


-----

## Navigation du Repository et Reproduction

### Organisation du Dépôt

```
.
├── Chatbot.ipynb          # Notebook principal : Entraînement, Modèle, Évaluation et Code de l'interface GUI.
├── presentation.pdf     # présentation (PDF recommandé)
├── github.pdf            # pdf github
├── notebook.pdf         # le fichier notebook (PDF recommandé)
├── data/                  # Contient les fichiers d'intentions.
├── image/                # Images et schémas (y compris l'image d'en-tête).
├── chat_model_tfidf.pth   # Modèle PyTorch sérialisé.
├── meta_tfidf.pkl         # Métadonnées du Vectoriseur et des Labels.
└── README.md              # Documentation du projet.
```

### Liens Utiles

| Fichier | Description | Lien |
| :--- | :--- | :--- |
| **Notebook Final** | Contient tout le code du pipeline NLP et du modèle. | **[`Chatbot.ipynb`](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/Chatbot.ipynb)** |
| **Notebook en format PDF** | Contient tout le code du pipeline NLP et du modèle en pdf. | **[`Chatbot.pdf`](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/notebook.pdf)** |
| **Présentation** | Le support visuel du projet. | **[`Voir la présentation`](https://github.com/Poincare008/Capstone-Chatbot-Intelligent/blob/main/powerpoint.pdf)** |
| **Licence** | Licence d'utilisation du projet. | **[`LICENSE`](https://www.google.com/search?q=/LICENSE)** |

### Instructions de Reproduction

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

**Auteur :** Antonine Pelicier & Steve Calixte

**Lien du Projet :** `https://github.com/Poincare008/Capstone-Chatbot-Intelligent`


[![Status](https://img.shields.io/badge/Status-Complet-%2300A8E8?style=flat-square)](https://github.com/Poincare008/Capstone-Chatbot-Intelligent)
[![Licence](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)
