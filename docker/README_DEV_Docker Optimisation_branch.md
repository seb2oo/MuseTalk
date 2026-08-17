cd D:\Dev\04_MuseTalkImprovement\MuseTalk
rm -rf /workspace/MuseTalk

1 : Utilsier le fichier "Dockerfile" pour créer une image avec docker-desktop par exemple :

Donc faire un git clone de https://github.com/seb2oo/MuseTalk sur son PC (pas besoin si on developpe deja en local "donc moi") et ensuite faire :
docker build -f docker/Dockerfile -t seb2oo/musetalk_projectv4 .
"""
docker build -f docker/Dockerfile -t (shouldBeYourUserNameOnDockerHub)/(shouldBeYourProjectNameOnDockerHub) .
ou
docker build --progress=plain -f docker/Dockerfile -t seb2oo/musetalk_projectv4 . --> Pour voir les étapes plus clairement
ps: si on modifie le dockerfile ou autre, pas besoin de supprimer l'image on va ré-ecrire dessus !
"""

2 : Pusher cette image sous docker-hub
docker push seb2oo/musetalk_projectv4:latest
ps : mettre un autre tag que latest si je repousse le projet V4 mais que je veux garder l'ancienne version V4 ! Sinon sera remplacé

3 : récuperer cette image avec RunPod et créer un pod avec ces paramètres là : 

(shouldBeYourUserNameOnDockerHub)/(shouldBeYourProjectNameOnDockerHub)
start command : 	sleep infinity
container disk size : 	30
volume disk size :	30 (c'est l'espace sur le "cloud", peut être 0 si nécessaire car runpod demander un minimum de crédit de 5 USD pour son utilisation)
volume mount path : 	/runpod-volume
Exposed HTTP ports : 	8000 , label : not forcely necessary

lancé sur GPU :
RTX A 5000


4 : Une fois dans le docker qui tourne, il faudra lancer ces commandes la : 

git clone --branch docker-optimisation https://github.com/seb2oo/MuseTalk.git /workspace/MuseTalk
pip install --no-cache-dir "huggingface_hub[cli]==0.30.2" --> A Faire au cas ou... car on ne las pas mit dans l'image docker de base ..
bash /workspace/MuseTalk/docker/setup_after_dockerRun.sh

**** 
CETTE PARTIE LA NE FAIT PAS PARTIE DE DOCKER SPéCIFIQUEMENT ! MAIS ELLE ME PERMET DE CHANGER RAPIDEMENT LES INPUTS (AUDIO,IMAGE,RéFéRENCE) AFIN DE FAIRE DES TEST! et je ne voulais pas créer une branche spécifique pour ce "petit" test

Attention, ici il faudra encore faire ces modifications la (en local): 
- Il faudra mettre dans ce répértoire "MuseTalk\data\audio", ceci : seb_audio.wav
- Il faudra mettre dans ce répértoire "MuseTalk\data\video", ceci : seb_image.png
- Il faudra mettre dans ce répértoire "MuseTalk\configs\inference\test.yaml", ceci : 
task_0:
 video_path: "data/video/seb_image.png"
 audio_path: "data/audio/seb_audio.wav"
- Il faudra aussi créer ce répértoire : "MuseTalk\results\test"
Attention ça fonctionne dans mon cas car le docker est créer avec mon github (setup_after_dockerRun.sh fait mon git clone) ! 

(changement locales envoyés sur gitHub)
git status
git add .
git status
git commit -m "changement d'inputs"
git push -u origin docker-optimisation

(récupération de ces changement depuis runpod)
cd /workspace/MuseTalk
git fetch origin
git checkout docker-optimisation
git pull origin docker-optimisation

(vérifications)
git branch
git status

cd MuseTalk
time python -m scripts.inference \
    --inference_config configs/inference/test.yaml \
    --result_dir results/test \
    --unet_model_path models/musetalkV15/unet.pth \
    --unet_config models/musetalkV15/musetalk.json \
    --version v15 \
    --use_png \
    --batch_size 8 \
    --use_float16


ci dessous la version qui ne charge que une fois les model lorsque l'on fait plusieurs inférences
ps : dans le cas d'un input d'une image, utiliser : use_png=True, use_saved_coord=False, saved_coord=False 
ps : dans le cas d'un input d'une vidéo, utiliser : use_png=False, use_saved_coord=False, saved_coord=True (pour la première inférence)
ps : dans le cas d'un input d'une vidéo, utiliser : use_png=False, use_saved_coord=True, saved_coord=True (pour TOUTES les prochaines inférences). Si on fait use_saved_coord=True, saved_coord=False, alors il va le supprimer...
ps : Dans la dernière version, use_png n'est même plus utilisé ! Car compatible directement pour image et vidéo
***
cd MuseTalk
python

import scripts.inference as inf # que une fois
from interactive import runtime # que une fois


args = inf.argparse.Namespace(
    inference_config="configs/inference/test.yaml",
    result_dir="results/test",

    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",

    version="v15",
    use_png=False,
    batch_size=8,
    use_float16=True,

    runtime=runtime,

    gpu_id=0,
    vae_type="sd-vae",
    whisper_dir="models/whisper",

    ffmpeg_path="",
    output_vid_name=None,

    fps=25,
    bbox_shift=0,

    audio_padding_length_left=2,
    audio_padding_length_right=2,

    left_cheek_width=90,
    right_cheek_width=90,

    extra_margin=10,
    parsing_mode="jaw",

    use_saved_coord=True,
    saved_coord=True,
)

inf.main(args)



exit()
***

le resultat pourra être visibile directement dans le serveur web ! (ca évite de faire un push dans github !)

python3 -m http.server 8000 --bind 0.0.0.0
ps : utiliser la console de runpod pour executer ce code, cela permet d'avoir une seconde console en plus du web terminal et ainsi lancer le 
server un parrallèle de mon kernel python qui tourne sous le web terminal


PS : ici un code exemple pour créer un tag une fois un milestone realisé :
git tag -a v0.1.0-speed-optimization-png -m "First speed optimization - static PNG input only"
git push origin v0.1.0-speed-optimization-png
****


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

***
use cherry pick instead of merge to get a special commit !
***