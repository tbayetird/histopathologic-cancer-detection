<div style="padding: 20px">
    
# <font color=purple>Présentation</font>

## <font color=purple>Contexte</font>

### 1. Présentation de Kaggle

<div style="text-align: justify; padding-top: 10px">
Communauté en ligne de Data Scientist et Machine Learners. Kaggle permet aux utilisateurs de rechercher et de publier des ensembles de données, d'explorer et de créer des modèles dans un environnement informatique basé sur la science des données, de travailler avec d'autres informaticiens et ingénieurs en apprentissage automatique et de participer à des concours pour relever les défis liés à la science des données.
</div>

### 2. Objectif Général

<div style="text-align: justify; padding-top: 10px">
En vue de mettre en pratique les connaissances acquises au cours des mois précédents nous nous proposons de participer au <a href="https://www.kaggle.com/c/histopathologic-cancer-detection">concours Kaggle</a> sur la Détection histopathologique du cancer: identification des tissus métastatiques sur des scans histopathologiques de sections de ganglions lymphatiques (Histopathologic Cancer Detection, Identify metastatic tissue in histopathologic scans of lymph node sections).
L'idée de ce concours sera de proposer et d'implémenter un model DeepLearning efficace pouvant atteindre un score de 90% avec des données préalablement fournies par Kaggle. 
<br/>
L'algorithme proposée devra pouvoir identifier des cancers métastatiques au niveau de portions d'images extraites de scans.
</div>

### 3. Définition des termes 

<div style="text-align: justify; padding-top: 10px">
<ul>
    <li><b>Histopathologie : </b> discipline médicale destinée à faire un diagnostic par l'étude microscopique de prélévements de tissus.
    </li>
    <li><b>Tissu : </b> Organization cellulaire ou ensemble de cellules</li>
    <li><b>Métastases : </b> Propagation d'un agent pathogène (cellules cancéreuses dans notre cas) d'un site primaire (cancer primaire) vers d'autres sites. Les nouveaux sites infectés seront également nommés Métastases.
    </li>
    <li><b>Tissus métastatiques : </b> Tissus abritant des métastases par conséquent tissus cancéreux</li>
</ul>
</div>

## <font color=purple>Données & Concours</font>

### 1. Données 

Les données proposées par le site Kaggle sont dérivées du dataset <a href="https://github.com/basveeling/pcam">PatchCamelyon (PCam)</a> dont les doublons ont été retiré. PCam est une base d'images riches cependant de petite taille contenant 327680 images en couleur de résolution (96 x 96px) extraites de scans histopathologiques de sections de ganglions lymphatiques. Vue sa taille, un model pourrai facilement exploiter ce dataset sur un GPU en quelques heures et obtenir des scores de detection de cancers assez elevés. Le jeu de données proposé comprend:
<ul>
    <li>une série d'images d'entrainement libellées <i>(fichier CSV)</i> 0 et 1 <i>(0 négatif et 1 positif)</i></li>
    <li>une série d'images de test</li>
</ul>

### 2. Concours

<ul>
    <li>Date de démarrage du concours: <b>16 Novembre 2018</b></li>
    <li>Date de fin du concours: <b>30 Mars 2019</b></li>
</ul>
<br/>
Les soumissions sont évaluées sur <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic">la surface sous la courbe ROC</a> entre la probabilité prédite et la cible observée.

## <font color=purple>Objectifs</font>

<ol>
    <li>Mettre en pratique les techniques de DeepLearning acquises</li>
    <li>Obtenir une première base de travail pouvant faire l'objet d'évolution. Cette base pourra être réutiliser dans le cadre d'autres travaux relatifs</li>
    <li>Implémenter un model performant et efficace avec un score minimum de 0.9 (90%) en un temps record</li>
</ol>

## <font color=purple>Méthodologie</font>

### 1. Cadre de travail

<ul>
    <li>Le code source sera stocké sur un <a href="https://github.com/tbayetird/histopathologic-cancer-detection">repository Github</a></li>
    <li>Le projet et tableau Kanbaan sera de même stocké sur Github</li>
</ul>

### 1. Architecture technique

L'implémentation de notre model se fera sur la base du framework DeepLearning Keras.
Le projet sera structuré ainsi:
<ul>
    <li><b>utils : </b> contient les utilitaires et outils de création rapide</li>
    <li><b>mock-data : </b> contient un sous ensemble du jeu de données afin de pouvoir tester rapidement le model</li>
    <li><b>models : </b> contient nos models et/ou models importés</li>
    <li><b>outputs : </b> contient le résultat des travaux</li>
    <li><b>config : </b> contient la configuration générale (chemins des fichiers et paramétres initiaux entre autres)</li>
</ul>
Nous commencerons avec un model de base sans convolutions puis nous ferons evoluer ce model en rajoutant différentes couches convolutionnelles. Au final afin de gagner en temps on pourrai utiliser un réseau pre-entrainé en vue de faire du transfert d'apprentissage (transfert learning) et de l'ajustement (fine tuning). 

</div>