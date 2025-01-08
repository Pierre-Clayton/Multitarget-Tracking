Projet dans le cadre de la 3ème année de l'ENSAE (Rémi Calvet, Pierre Clayton) et du M2 DS (Nicolas Alemany).

# Multitarget-Tracking
On MCMC-Based Particle Methods for Bayesian Filtering: Application to Multitarget Tracking


Le script tentative_1.py constituait notre première version. Il permettait de simuler les trajectoires, de générer des observations et de tenter de prédire les trajectoires des cibles. Nous avons obtenu des prédictions, mais celles-ci ne couvraient pas toutes les cibles. Nous nous sommes rendu compte que l'algorithme utilisé pour faire ces prédictions dans ce script n'était pas exactement celui décrit par les auteurs dans l'article.

C'est pourquoi nous sommes repartis de zéro avec le script HMM_last_version_algo afin d'essayer de mieux implémenter l'algorithme décrit par les auteurs. Nous pensons que l'implémentation dans HMM_last_version_algo est plus proche de celle des auteurs, mais nous n'avons pas réussi à obtenir de bonnes prédictions des trajectoires.

L'autre branche du repository contient toutes nos tentatives et travail effectué.
