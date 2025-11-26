"""
main.py
Point d'entrée principal du pipeline ASR :
- Téléchargement du dataset
- Inférence du modèle de base
- Entraînement des adaptateurs
- Inférence du modèle fine-tuné
- Calcul du WER final

Auteur : Ngom Christine
Projet : ASR Fellowship Challenge – Adapter Fine-Tuning
"""

from src.utils.data_utils import download_dataset
from src.inference.inference_base import run_inference as run_base
from src.training.train import train
from src.inference.inference_finetuned import run_inference as run_finetuned
from src.evaluation.evaluate import compute_wer

def main():

    print("\n=== Étape 1 : Téléchargement du dataset ===")
    download_dataset()

    print("\n=== Étape 2 : Inférence du modèle de base ===")
    run_base("data/test", "base_transcriptions.txt")

    print("\n=== Étape 3 : Entraînement des adaptateurs ===")
    train()

    print("\n=== Étape 4 : Inférence du modèle fine-tuné ===")
    run_finetuned("data/test", "finetuned_transcriptions.txt")

    print("\n=== Étape 5 : Évaluation (WER) ===")
    compute_wer(
        "data/test/references.txt",
        "finetuned_transcriptions.txt"
    )

    print("\n=== Pipeline terminé avec succès ===")

if __name__ == "__main__":
    main()
