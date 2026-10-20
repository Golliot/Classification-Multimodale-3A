# Installation et utilisation

## Contenu du dépôt

| Versionné | Non versionné (`.gitignore`) |
|---|---|
| `scripts/` : téléchargement et pré-traitement du dataset en 1FPS | `data/` : dataset pré-traité (~172 Go), à se procurer à partir de prepare_vggsound.py |
| `concatenation_all_frames.ipynb` : modèle, entraînement, évaluation | `temp_data/` : archives temporaires du pré-traitement (vide une fois le téléchargement et prétraitement terminés) |
| `readme.md`, `requirements.txt` | `checkpoints/`, `*.pth` : poids des modèles |
| | `logs/`, `.venv/`, `__pycache__/` |

## Prérequis

- **GPU NVIDIA** avec un pilote récent (≥ 580, pour les wheels PyTorch CUDA 13). Le notebook a été écrit pour 16 Go de VRAM : avec moins, baisser `MICRO_BATCH_SIZE` (cellule des DataLoaders).
- Si GPU plus ancien, il faudra désactiver certaines optimisations telles que le channels_last, l'AMP et le TF32.
- **~200 Go d'espace disque** pour le dataset (SSD fortement conseillé : chaque époque relit ~1,8 million d'images).
- **Python 3.14** (version testée). Le plus simple est de passer par [uv](https://docs.astral.sh/uv/), qui télécharge lui-même la bonne version de Python.
- **git** et **curl** (curl est fourni avec Windows 11).
- Un éditeur capable d'exécuter un notebook : VS Code + extension Jupyter, ou JupyterLab (`pip install jupyterlab`).

## Ubuntu

### 1. Récupérer le projet et créer l'environnement

```bash
git clone <url-du-depot> classification_multimodale
cd classification_multimodale

# Avec uv (recommandé)
curl -LsSf https://astral.sh/uv/install.sh | sh        # si uv n'est pas installé
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# ... ou avec pip, si python3.14 est déjà installé
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Vérification : `python -c "import torch; print(torch.cuda.is_available())"` doit afficher `True`.

### 2. Préparer le dataset

Toutes les commandes se lancent depuis la racine du projet, environnement activé.

```bash
# Téléchargement (HuggingFace : Loie/VGGSound, 20 archives) et pré-traitement :
# 1 image JPEG par seconde + audio FLAC mono 16 kHz. Long (plusieurs heures) mais reprenable :
# relancer la même commande après une interruption reprend là où elle s'était arrêtée.
python -I scripts/prepare_vggsound.py                  # tout (= --subsets 0-19)
python -I scripts/prepare_vggsound.py --subsets 0-2    # ou seulement quelques archives pour tester

# Création du split val (5 % de train, stratifié par classe), une fois le pré-traitement terminé
python -I scripts/make_val_split.py

# Étiquettes corrigées du test (VGGSounder, Apache 2.0), utilisées par les dernières cellules du notebook
curl -L -o data/vggsounder_test_labels.csv \
  "https://raw.githubusercontent.com/Bizilizi/VGGSounder/5a91bf6/supplimentary/data/vggsounder%2Bbackground-music.csv"
```

À la fin, `data/` contient `metadata.csv`, `vggsound.csv`, `vggsounder_test_labels.csv`, `manifests/` et `{train,val,test}/{frames,audio}/`.

### 3. Lancer le notebook

Ouvrir `concatenation_all_frames.ipynb` dans VS Code et choisir le noyau `.venv` (ou lancer `jupyter lab` depuis l'environnement), puis exécuter les cellules dans l'ordre.

- **Reprise** : si `checkpoints/checkpoint_all_frames.pth` existe, l'entraînement reprend depuis ce checkpoint ; sinon il démarre de zéro.
- **`torch.compile`** (`USE_COMPILE`, cellule des hyperparamètres) : nécessite les en-têtes de Python et un compilateur C (`sudo apt install python3.14-dev build-essential` si Python vient d'apt). En cas d'erreur à la compilation, passer `USE_COMPILE = False`.
- **`NUM_WORKERS`** : réglé à 10 pour un CPU à 12 threads. Le ramener à (nombre de threads − 2) sur une machine plus petite.

## Windows 11

### Option A (recommandée) : WSL2

Les scripts et le notebook sont écrits pour Linux ; sous WSL2, ils fonctionnent sans aucune modification, GPU compris.

1. Installer le **pilote NVIDIA pour Windows** uniquement (ne pas installer de pilote dans WSL : le GPU y est exposé automatiquement).
2. Dans un PowerShell administrateur :
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   puis redémarrer et créer l'utilisateur Ubuntu.
3. Dans le terminal Ubuntu, vérifier le GPU avec `nvidia-smi`, puis suivre **exactement** la section [Ubuntu](#ubuntu) ci-dessus.
4. **Cloner le projet dans le système de fichiers Linux** (par ex. `~/classification_multimodale`), pas sous `/mnt/c/...` : l'accès aux disques Windows depuis WSL est beaucoup trop lent pour lire des millions d'images. Vérifier que le disque virtuel de WSL dispose d'assez d'espace (~200 Go).
5. Pour éditer et exécuter le notebook : VS Code sous Windows + extension **WSL**, puis `code .` depuis le dossier du projet dans le terminal Ubuntu.

### Option B : Windows natif (limité)

- **Pré-traitement : non supporté tel quel.** Les scripts remplacent des fichiers existants avec `Path.rename`, ce qui échoue sous Windows (`FileExistsError`) dès que `data/metadata.csv` ou les manifestes sont réécrits. Faire le pré-traitement sous WSL2 ou Linux, ou récupérer le dossier `data/` d'un membre du groupe.
- **Notebook : possible**, avec les étapes ci-dessous.

Installation (PowerShell, depuis la racine du projet) :

```powershell
# uv : https://docs.astral.sh/uv/getting-started/installation/
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv venv --python 3.14 .venv
.venv\Scripts\activate

# PyTorch CUDA d'abord : la version PyPI de torch pour Windows ne gère que le CPU
uv pip install torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Si une version n'est pas disponible sur l'index CUDA, prendre la commande proposée par le sélecteur de [pytorch.org](https://pytorch.org/get-started/locally/) (Windows, Pip, CUDA 13).

Le fichier VGGSounder se télécharge de la même façon qu'en section Ubuntu (`curl.exe -L -o data\vggsounder_test_labels.csv "<url>"`).

Avant d'exécuter le notebook, trois réglages locaux sont nécessaires :

1. **Première cellule** : commenter `mp.set_start_method('fork', force=True)` (la méthode `fork` n'existe pas sous Windows).
2. **Cellule des DataLoaders** : `NUM_WORKERS = 0`. Sous Windows, les workers ne peuvent pas ré-importer les classes définies dans le notebook. Le chargement des données se fait alors dans le processus principal : l'entraînement sera nettement plus lent (le CPU devient le facteur limitant).
3. **Cellule des hyperparamètres** : `USE_COMPILE = False` (Triton, utilisé par `torch.compile`, n'est pas supporté sous Windows).

Ces modifications sont propres à votre machine : **ne pas les committer** (vérifier `git diff` avant chaque commit).

## Partage des poids entraînés

Les fichiers `.pth` ne sont pas versionnés (~400 Mo). Pour réutiliser un modèle entraîné par un autre membre du groupe, le récupérer par un autre canal (Drive, WeTransfer, etc...) et le placer dans `checkpoints/` :

- `best_macro_f1_all_frames.pth` : poids seuls du meilleur modèle (macro-F1 de validation), chargés par les cellules d'évaluation sur le test ;
- `checkpoint_all_frames.pth` : poids + optimiseur + historique, pour reprendre l'entraînement.










# Les implémentations



Dataset: On se base sur un VGG sound (video + audio à 309 classes) réduit pour nos besoins. Celui-ci n'a, de base, qu'un split train/test. On a divisé le set train en train/val (ratio 95:5 respectant la répartition des classes)

On propose plusieurs approches. En premier lieu, on tente une late fusion : deux encodeurs travaillent de façon indépendante, puis leurs features sont normalisées puis passées à la tête de classification.

# Late fusion

## Implémentation naïve : 10 frames par vidéo, concatenation_all_frames.ipynb

Par faute de moyens, on a effectué la réduction suivante sur le dataset. Dans chaque vidéo, on sélectionne uniformément une frame par second, que l'on convertit en 2565x256 px. Le son est converti en flac 16khz. A chaque epoch, on fait passer le spectrogramme et les 10 frames d'un coup (on les passe à la suite dans le CNN Image)

Modèle:

Le CNN image est un ResNet34 (skip connections longues et courtes) entraîné from scratch. Image -> Embedding (512)

Pour la partie audio, on convertit les audios en spectrogramme log-mel à 64 bandes, avant de les passer à un CNN audio basique. Le CNN lui-même est un petit Resnet18. Son -> Embedding (512)

Normalisation, concaténation des features, puis tête de classification de type MLP 1024 -> 309

Pour l'entraînement lui-même :
On a introduit des têtes de classification auxiliaires à l'audio et à l'image, avec leur propre loss. On pénalise davantage l'image, car sinon le modèlme a tendance à se concentrer sur l'audio, qui lui donne plus d'onformation.

### Résultats et diagnostic:

Par construction, un spectrogramme contient beaucoup plus d'informations qu'une image (il n'est pas sujet à des translations, rotations, etc...). En plus de ça, on s'est limités à seulement 1 FPS, ce qui est très peu pour de l'analyse vidéo. On a pu atténuer cet écart en ajoutant des loss auxiliaires, ainsi que du modality dropout sur le son, mais ça ne suffit pas : notre dataset est trop pauvre en images. En plus de ça, notons que le CNN image est bien plus complexe que le CNN audio (21.8M vs 3.0M paramètres)

On arrive malgré tout à une précision top-1 de 0.55 et une top-5 de 0.78 en phase de test. Les têtes unimodales, elles, ont une précision top-1 de 0.31 pour l'image et 0.47 pour l'audio.

Par la suite, on essaiera de conserver plus de frames, quitte à baisser la révolution. On aura alors 2 possibilités :
    1 - passage à un CNN vidéo 3D. Avec suffisament de frames, il pourrait prendre en compte les variations temporelles de la vidéo, plutôt que d'apprendre sur des frames a priori indépendantes.
    2 - plutôt que de se contenter de late fusion, on pourra essayer d'introduire des fusions intermédiaires basées sur de la cross-modal attention. Reste à déterminer l'architecture exacte.
On utilisera un CNN vidéo/Image pré-entraîné pour la suite.

Malgré ses nombreux défauts, cette première implémentation nous sert de base solide, ainsi que de point de référence pour de futures améliorations.

---

