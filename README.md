# Image Classifier avec CNN sur CIFAR-10

Ce projet est un petit classificateur d’images entraîné sur le dataset CIFAR-10, construit avec un réseau de neurones convolutifs basé sur le CNN. L’application finale est présentée avec Streamlit.

##Les fichiers importants

- `app.py` : fichier de l’application Streamlit
- `ImageClassifier.ipynb` : le notebook du code de l'entrainement
- `cnn_model_cifar10.keras` : fichier du modèle entrainé
- `images_inedites_test/` : dossier contenant des images inédites téléchargés sur google
- `requirements.txt` : la liste des dépendances à installer

##Comment lancé l'application streamlit:
- Une fois les dépendances installées, lance l'application avec la ligne de commande:

  ```bash ou dans le terminal
streamlit run app.py
