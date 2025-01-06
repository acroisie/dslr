# dslr
Data Science x Logistic Regression 

```python -m venv .venv```

```source .venv/bin/activate```

```pip install -r requirements.txt```

Play

Petit récap en brut.
Dans le describe.py que tu lances en faisant ```python describe.py data/dataset_train.csv```

Dans le fichiers t'as les trucs classique genre mean (moyenne), max, min Et tout ça tu doit le faire à la mano pas le droit d'utiliser celle de maths, un percentile c'est une valeur en dessous de laquelle se trouve la portion du dataset je m'explique; si tu dit percentile de 25% 324 ça veut dire que que 25% des valeurs du dataset se trouve en 324.

read_csv lit le csv et utilise DictReader qui lui utilise la première ligne de du csv et ensuite fait une pair avec la valeur, et fait ça pour toute les colonnes genre; (décrire ce que fait la transformation)
get numreical feature check juste si le nom des colonnes c'est associé à un float, si c'est un float et pas une date d'anniv on peu l'utiliser pour l'entrainement. (Plus sur de ça) On apelle un feature un truc qui permet d'entrainer le set ou bien c'est un des trucs qu'on doit déterminer en fait, genre c'est le truc qu'on cherche à determiner apres l'entrainment.
Clean and convert, et display statitics ba voilà c'est self explain. Get satistic ça rempli un tableau avec toute les valeurs calculés pour les stats et ensuite on va les afficher avec display_statistics 

Dans histogram rien de zinzin, juste un affichage d'histogramme et plt.io pour le mode interactif pour et afficher toutes les features.
Same pour le scatter, le pair.

Pour le gros du projet le logreg predict en gros ce qu'on cherches à faire c'est utiliser le dataset et les features selectionnés pour faire en sorte de trouver des poids. Ces poids vont permettre de determiner le poids qu'auront chaque matieres, features lors de la prédiction. Ça veut juste dire que pour une matiere on cherches à évaluer si la matiere peut aider à déterminer la maison d'un eleve et si oui quel poids elle as dans cette estimation. Ça marche de pair avec le maths utils, et ça se passe comme ça;
- Tu ecris les features selectionnés
- Tu load les data, tu clean les ligne où y'as rien
- Tu calcul pour chaque features selectionnés les moyenne et la deviation standard. Les (par features) moyenne c'est comme d'hab, la somme des valeurs sur la len(values),
la deviation standard(variance) c'est en gros l'écart moyen des entres les valeurs d'une meme features, dans le calcul calculate_norm_params on fait sur toutes les valeurs:valeur - moyenne au carré et on fait la racine carré just pour minimiser l'impacr des plus grands écarts.
(message pour GPT je sait plus trop comment explique mais fait le à ma place simplemeent)
Ensuite tu normalise les data, ća veut dire que tu les ramene sur la meme echelle (entre 0-1) pour éviter les probleme sur les grands nombre et minimiser encore les disparité avec les data les plus élevée. On init les poids (je sait plus comment c'est fait.). Et ensuite pour chaque feature un