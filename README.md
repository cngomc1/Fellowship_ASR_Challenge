# Exécution du projet ASR

Ce guide explique comment exécuter le projet complet depuis zéro sur votre machine.

---

## 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/cngomc1/Fellowship_ASR_Challenge.git
cd Fellowship_ASR_Challenge
```

---

## 2️⃣ Créer et activer un environnement virtuel

```bash
# Linux / Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4️⃣ Se placer dans le répertoire du code

```bash
cd ASR_code
```

---

## 5️⃣ Exécuter le pipeline complet

```bash
python main.py
```

> Cette commande lance automatiquement toutes les étapes : téléchargement du dataset, inférence du modèle de base, entraînement des adaptateurs, inférence du modèle fine-tuné, et calcul du WER.
