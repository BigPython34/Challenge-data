#!/usr/bin/env python3
"""Script pour vérifier le modèle actuel"""
import pickle

with open("trained_models/model_package.pkl", "rb") as f:
    package = pickle.load(f)

print("Meilleur modèle:", package["best_model_name"])
print("Type:", type(package["best_model"]["model"]))
print("Score:", package["best_model"]["score"])
