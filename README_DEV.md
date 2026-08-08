***
Base project :
MuseTalk
***

***
Official repository :
https://github.com/TMElyralab/MuseTalk
***

***
Fork :
https://github.com/seb2oo/MuseTalk
***

***
Start Version :

git rev-parse HEAD
0a89dec45a0192b824e3cf4daf96c239440c5ed8

git log -1
commit 0a89dec45a0192b824e3cf4daf96c239440c5ed8 (HEAD -> main, upstream/main, origin/main, origin/HEAD)
Author: 李洋 <liyangmsn@live.com>
Date:   Fri Sep 26 13:44:17 2025 +0800

    feat: update download_weights.bat (#372)

    更换weights下载工具
    更换face-parse-bisent源

git describe --tags --always
0a89dec

git tag

git log --oneline -10
0a89dec (HEAD -> main, upstream/main, origin/main, origin/HEAD) feat: update download_weights.bat (#372)
8c19579 fix: convert all audio to WAV 16kHz PCM before processing (#379)
9deb9be fix: ensure upper bond does not go below zero in landmark extraction (#329)
6e39bd0 fix: preprocess import bug (#345)
26ca7c2 fix: use torch.no_grad() in inference to prevent excessive memory usage (~30GB) with inference (#349)
8ca7d18 fix: download_weights.sh (#318)
67e7ee3 feat: windows infer & gradio (#312)
36163fc Update audio_processor.py
7b829ba docs: update readme
1ab53a6 feat: data preprocessing and training (#294)
***

***
Date :
07 august 2026
***

***
Objectives :

- Docker Optimisation
- Gradio interface 
- inference spped improvements
- parameters testing
- Tuning
***


DONE (FOR ONE BRANCH (here is : docker), IS GONNA BE THE SAME WAY FOR OTHER BRANCH) : 
MuseTalk/    
docker/
    │
    ├── Dockerfile
    ├── setup_after_dockerRun.sh
    ├── download_models.sh
    ├── verify_install.sh
    └── README_DEV_Docker Optimisation_branch.md
otherBranch
    │
    ├── ...
    ├── ...
    ├── ...
    ├── ...
    └── README.md
etc..


0 :
git add README_DEV.md
git commit -m "Add development documentation"
git push -u origin main (use this only for the first time : -u origin main in order to "point" the current branch we are working one...)

ps: This readme is evolving with the project, sometimes it should be push with commit "updateReadme" but is not forcely necessary

1 : Créer le dossier plus haut et ses fichiers internes !

2 :
git checkout -b "docker-optimisation"
git commit --allow-empty -m "Create docker-optimisation branch"

3 :
git add docker/
git commit -m "Add docker setup empty files"
git push -u origin docker-optimisation

ps : here exemple for single file : git add docker/README_DEV_Docker Optimisation_branch.md

X:
look the projet commit and label to understand what was done ! 



AT THE END : MAKE THIS README ON THE MARDOWN FORMAT