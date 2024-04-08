#!/bin/zsh

# Naviguer vers le répertoire de votre projet
cd /home/pidoux/LIMICS/EDS-RENE || exit

# Trouver et synchroniser tous les fichiers .ipynb avec Jupytext
for notebook in **/*.ipynb(.); do
    # Synchroniser le .ipynb avec .md (crée le .md s'il n'existe pas)
    jupytext --to markdown --set-formats ipynb,md $notebook
done

echo "Tous les fichiers .ipynb ont été synchronisés avec leurs versions .md."

# Exécuter Flake8 sur tous les fichiers Python dans le répertoire courant et sous-répertoires
flake8 .

echo "Flake8 a terminé de vérifier les fichiers Python."

# Exécuter Pre-commit sur tous les fichiers pour vérifier les autres hooks configurés
pre-commit run --all-files

echo "Pre-commit a terminé de vérifier les fichiers."
