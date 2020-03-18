# DRL_class

## Commandes

Pour activer l'environnement virtuel :

```
pipenv shell (a faire dans le dossier)
```

Pour installer les libs :

```
pip install -r .\requirements.txt
```

Pour lancer un entraînement :

```
python Training.py model
```
où model est le nom du modèle qui va être sauvegardé (il sera précédé de "model" par défaut)
Par exemple, pour créer model_0 :
```
python Training.py _0
```

Pour lancer les tests sur le modèle :

```
python test.py model nb_render
```
où model est le nom du modèle qui va être testé (il sera précédé de "model" par défaut) et nb_render le nombre d'épisodes dont on veut le rendu à la fin (par défaut 1)
Par exemple, pour tester model_0 sans rendu à la fin :
```
python test.py _0 0
```
(sont déjà presents les modèles model_1 (qui obtient des scores généralement élevés) et model_2 et model_3 (qui perdent en faisant tomber le baton vers la droite)
