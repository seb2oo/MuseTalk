1 : Utilsier le fichier "Dockerfile" pour créer une image avec docker-desktop par exemple :

Donc faire un git clone de https://github.com/seb2oo/MuseTalk sur son PC et ensuite faire :
docker build -f docker/Dockerfile -t seb2oo/musetalk_projectv4 .
"""
docker build -f docker/Dockerfile -t (shouldBeYourUserNameOnDockerHub)/(shouldBeYourProjectNameOnDockerHub) .
"""

2 : Pusher cette image sous docker-hub
docker push seb2oo/musetalk_projectv4:latest

3 : récuperer cette image avec RunPod et créer un pod avec ces paramètres là : 

(shouldBeYourUserNameOnDockerHub)/(shouldBeYourProjectNameOnDockerHub)
start command : 	sleep infinity
container disk size : 	30
volume disk size :	30
volume mount path : 	/runpod-volume
Exposed HTTP ports : 	8000 , label : not forcely necessary

lancé sur GPU :
RTX A 5000


4 : Une fois dans le docker qui tourne, il faudra lancer ces commandes la : 
- bash /workspace/setup_after_dockerRun.sh
- cd /workspace/MuseTalk

time python -m scripts.inference \
    --inference_config configs/inference/test.yaml \
    --result_dir results/test \
    --unet_model_path models/musetalkV15/unet.pth \
    --unet_config models/musetalkV15/musetalk.json \
    --version v15

***
Et par la suite si je modifie ceci :

docker/Dockerfile
docker/setup_after_dockerRun.sh
docker/download_models.sh
docker/verify_install.sh

puis:

git add docker/
git commit -m "Improve Docker setup"
git push origin docker-optimisation

sur runpod je peux faire ceci :
cd /workspace/MuseTalk
git pull origin docker-optimisation

et tester directement.

PS : Ceci est un exemple car cette branch sert à la mise en place du docker et une fois fonctionnelle, on ne fait plus aucun changement.
Ce sera par exemple la branch tuning qui necessitera des changement comme par exemple (chagement d'inputs ou autres)
***
