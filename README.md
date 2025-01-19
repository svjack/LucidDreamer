<p align="center">
    <img src="assets/logo_color.png" height=180>
</p>

# 😴 LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes 😴

<div align="center">

[![Project](https://img.shields.io/badge/Project-LucidDreamer-green)](https://luciddreamer-cvlab.github.io/)
[![ArXiv](https://img.shields.io/badge/Arxiv-2311.13384-red)](https://arxiv.org/abs/2311.13384)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ironjr/LucidDreamer-mini)

</div>

<div align="center">

[![Github](https://img.shields.io/github/stars/luciddreamer-cvlab/LucidDreamer)](https://github.com/luciddreamer-cvlab/LucidDreamer)
[![X](https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![LICENSE](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/LICENSE)

</div>


https://github.com/luciddreamer-cvlab/LucidDreamer/assets/12259041/35004aaa-dffc-4133-b15a-05224e68b91e


> #### [LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2311.13384)
> ##### \*[Jaeyoung Chung](https://robot0321.github.io/), \*[Suyoung Lee](https://esw0116.github.io/), [Hyeongjin Nam](https://hygenie1228.github.io/), [Jaerin Lee](http://jaerinlee.com/), [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)
###### \*Denotes equal contribution.

<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>

---

- https://huggingface.co/spaces/svjack/LucidDreamer
- app.py
```bash
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg
conda create -n py39 python=3.9 && conda activate py39
pip install ipykernel
python -m ipykernel install --user --name py39 --display-name "py39"

git clone https://huggingface.co/spaces/svjack/LucidDreamer && cd LucidDreamer
pip install -r requirements.txt
pip install numpy==1.26.0

### choose don't use to try oneselves demo
python app.py
```

- https://huggingface.co/spaces/svjack/LucidDreamer
- genshin_impact_couple_app.py
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('svjack/Genshin-Impact-Couple-with-Tags-IID-Gender-Only-Two-Joy-Caption', split='train[:10]')

# Define the short captions list
short_l = [
    "Anime-style illustration of two 'Genshin Impact' characters in a cozy tea house: a woman in a white kimono with purple eyes and a boy in a sailor outfit with blue hair, surrounded by warm lighting and tea items.",
    "Anime-style scene of a cheerful blonde girl in an orange skirt and a formal blue-haired boy walking hand-in-hand on a sunny, tree-lined street with shops and a lively atmosphere.",
    "Anime-style illustration of a red-haired woman in a white top and blue jeans sitting with a teal-streaked man in a park, under a sunny sky with trees and flowers.",
    "Anime-style image of two characters in a cozy restaurant: a blue-haired man in a hoodie and a green-haired woman in a robe, sharing food at a table with warm lantern light.",
    "Anime-style kitchen scene with a brown-haired man in a yellow shirt and a blonde woman in a maid outfit holding a cake, in a warm, sunlit kitchen filled with dishes.",
    "Anime-style illustration of two women dining in a traditional Japanese restaurant: a blonde in an off-white top and a purple-haired woman in a lace-adorned top, enjoying a meal in a warm, wooden setting.",
    "Anime-style kitchen scene with a blue-haired man in a polo shirt and a brown-haired man in an apron preparing a roasted turkey, in a bright, modern kitchen.",
    "Anime-style outdoor scene of two characters in a park: a blue-haired figure in a traditional outfit and a green-haired figure in a robe, sharing tea and cake under soft natural light.",
    "Anime-style illustration of a teal-haired man in a sleeveless shirt and a purple-haired woman in a gold dress sitting at a wooden table in a cozy, warm-toned interior.",
    "Anime-style living room scene with a green-haired man taking a selfie and a magenta-haired woman holding cards, in a well-lit room with a desk, plants, and bookshelves."
]

# Define the background descriptions list (joy_caption_surrounding)
surrounding_l = [
    "A cozy tea house with a wooden interior, warm lighting, and tea items like a teapot, cups, and a tray on the counter.",
    "A sunny, tree-lined street with shops, including a bakery and bookstore, and a lively atmosphere with street lamps and people.",
    "A serene park with tall trees, dappled shadows on lush green grass, and small white flowers under a bright, sunny sky.",
    "A warm, cozy restaurant with a wooden table, a candle in a red holder, and softly blurred background featuring other diners and hanging lanterns.",
    "A warm, sunlit kitchen with wooden cabinets, shelves filled with dishes, and a bowl of fruit and sliced fruit on the counter.",
    "A traditional Japanese-style restaurant with wooden walls, sliding doors, and soft lighting from hanging lamps, creating a calm and inviting ambiance.",
    "A bright, modern kitchen with white tiled walls, stainless steel appliances, and a large window letting in natural light.",
    "A lush green park with tall trees, a winding path, and a small white table with a brown wooden chair under soft, natural light.",
    "A cozy, warm-toned interior with wooden walls, soft lighting, and a small bowl of soup or stew on the table.",
    "A well-lit modern living room with a wooden desk, a potted plant, a bookshelf filled with books, and soft gray walls with natural light streaming in."
]

# Define a function to add the `joy_caption_short` and `joy_caption_surrounding` columns
def add_captions(example, index):
    example["joy_caption_short"] = short_l[index]
    example["joy_caption_surrounding"] = surrounding_l[index]
    return example

# Apply the function to the dataset using map
updated_dataset = dataset.map(add_captions, with_indices=True)

# Save the updated dataset to disk
updated_dataset.save_to_disk("Genshin-Impact-Couple-with-Tags-IID-Gender-Only-Two-Joy-Caption_Head10")
```

```bash
git clone https://huggingface.co/spaces/svjack/LucidDreamer Gen_LucidDreamer && cd Gen_LucidDreamer
python genshin_impact_couple_app.py
```

- https://github.com/svjack/LucidDreamer
- run.py
```bash
conda create -n py39 python=3.9 && conda activate py39
pip install ipykernel
python -m ipykernel install --user --name py39 --display-name "py39"

git clone https://github.com/svjack/LucidDreamer && cd LucidDreamer
pip install -r gradio_requirements.txt 
pip install numpy==1.26.0

python run.py
``` 

```python
import pathlib 
import pandas as pd


def r_func(x):
    with open(x, "r") as f:
        return f.read().strip()

anime_jpg_l = pd.Series(list(pathlib.Path("examples/").rglob("*anime*.jpg"))).map(str)

print("\n\n".join(anime_jpg_l.map(
    lambda x: (x, r_func
               (x.replace(".jpg", ".txt")))
).map(
    lambda t2: 'python run.py --image "{}" --text "{}"'.format(t2[0], t2[1])
).values.tolist()))
```

```bash
python run.py --image "examples/Image018_animesummerhome.jpg" --text "Anime-style, Japanese-style anime house overlooking the anime sea with anime tatami mats, anime curtains blowing in the wind, anme clouds visible in the anime sky, anime livingroom with anime flowers"

python run.py --image "examples/Image015_animelakehouse.jpg" --text "anime style, animation, best quality, a boat on lake, trees and rocks near the lake. a house and port in front of a house"

python run.py --image "examples/Image014_animestreet.jpg" --text "best quality, 4k, anime-style, anime, manga style, a long anime-style road with anime-blocks and little anime-grass, anime-houses and anime-tree on the side of the anime-style road, wide anime-style bright blue sky, shiny and beautiful day, bright scene"
```


## 🤖 Install

### Ubuntu

#### Prerequisite

- CUDA>=11.4 (higher version is OK).
- Python==3.9 (cannot use 3.10 due to open3d compatibility)

#### Installation script

```bash
conda create -n lucid python=3.9
conda activate lucid
pip install peft diffusers scipy numpy imageio[ffmpeg] opencv-python Pillow open3d torch==2.0.1  torchvision==0.15.2 gradio omegaconf
# ZoeDepth
pip install timm==0.6.7
# Gaussian splatting
pip install plyfile==0.8.1

cd submodules/depth-diff-gaussian-rasterization-min
# sudo apt-get install libglm-dev # may be required for the compilation.
python setup.py install
cd ../simple-knn
python setup.py install
cd ../..
```

### Windows (Experimental, Tested on Windows 11 with VS2022)

#### Checklist

- Make sure that the versions of your installed [**CUDA**](https://developer.nvidia.com/cuda-11-8-0-download-archive), [**cudatoolkit**](https://anaconda.org/nvidia/cudatoolkit), and [**pytorch**](https://pytorch.org/get-started/previous-versions/) match. We have tested on CUDA==11.8.
- Make sure you download and install C++ (>=14) from the [Visual Studio build tools](https://visualstudio.microsoft.com/downloads/).

#### Installation script

```bash
conda create -n lucid python=3.9
conda activate lucid
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install peft diffusers scipy numpy imageio[ffmpeg] opencv-python Pillow open3d gradio omegaconf
# ZoeDepth
pip install timm==0.6.7
# Gaussian splatting
pip install plyfile==0.8.1

# There is an issue with whl file so please manually install the module now.
cd submodules\depth-diff-gaussian-rasterization-min\third_party
git clone https://github.com/g-truc/glm.git
cd ..\
python setup.py install
cd ..\simple-knn
python setup.py install
cd ..\..
```

## ⚡ Usage

We offer several ways to interact with LucidDreamer:

1. A demo is available on [`ironjr/LucidDreamer` HuggingFace Space](https://huggingface.co/spaces/ironjr/LucidDreamer) (including custom SD ckpt) and [`ironjr/LucidDreamer-mini` HuggingFace Space](https://huggingface.co/spaces/ironjr/LucidDreamer-mini) (minimal features / try at here in case of the former is down)
(We appreciate all the HF / Gradio team for their support).

https://github.com/luciddreamer-cvlab/LucidDreamer/assets/12259041/745bfc46-8215-4db2-80d5-4825e91316bc

2. Another demo is available on a [Colab](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb), implemented by [@camenduru](https://github.com/camenduru)
(We greatly thank [@camenduru](https://github.com/camenduru) for the contribution).
3. You can use the gradio demo locally by running [`CUDA_VISIBLE_DEVICES=0 python app.py`](app.py) (full feature including huggingface model download, requires ~15GB) or [`CUDA_VISIBLE_DEVICES=0 python app_mini.py`](app_mini.py) (minimum viable demo, uses only SD1.5).
4. You can also run this with command line interface as described below.

### Run with your own samples

```bash
# Default Example
python run.py --image <path_to_image> --text <path_to_text_file> [Other options]
``` 
- Replace <path_to_image> and <path_to_text_file> with the paths to your image and text files.

#### Other options
- `--image` (`-img`): Specify the path to the input image for scene generation.
- `--text` (`-t`): Path to the text file containing the prompt that guides the scene generation.
- `--neg_text` (`-nt`): Optional. A negative text prompt to refine and constrain the scene generation.
- `--campath_gen` (`-cg`): Choose a camera path for scene generation (options: `lookdown`, `lookaround`, `rotate360`).
- `--campath_render` (`-cr`): Select a camera path for video rendering (options: `back_and_forth`, `llff`, `headbanging`).
- `--model_name`: Optional. Name of the inpainting model used for dreaming. Leave blank for default(SD 1.5).
- `--seed`: Set a seed value for reproducibility in the inpainting process.
- `--diff_steps`: Number of steps to perform in the inpainting process.
- `--save_dir` (`-s`): Directory to save the generated scenes and videos. Specify to organize outputs.


### Guideline for the prompting / Troubleshoot

#### General guides

1. If your image is indoors with specific scene (and possible character in it), you can **just put the most simplest representation of the scene first**, like a cozy livingroom for christmas, or a dark garage, etc. Please avoid prompts like 1girl because it will generate many humans for each inpainting task.
2. If you want to start from already hard-engineered image from e.g., StableDiffusion model, or a photo taken from other sources, you can try **using [WD14 tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) ([huggingface demo](https://huggingface.co/spaces/deepghs/wd14_tagging_online)) to extract the danbooru tags from an image**. Please ensure you remove some comma separated tags if you don't want them to appear multiple times. This include human-related objects, e.g., 1girl, white shirt, boots, smiling face, red eyes, etc. Make sure to specify the objects you want to have multiples of them.

#### Q. I generate unwanted objects everywhere, e.g., photo frames.

1. Manipulate negative prompts to set harder constraints for the frame object. You may try adding tags like twitter thumbnail, profile image, instagram image, watermark, text to the negative prompt. In fact, negative prompts are the best thing to try if you want some things not to be appeared in the resulting image.
2. Try using other custom checkpoint models, which employs different pipeline methods: LaMa inpainting -> ControlNet-inpaint guided image inpainting.

### Visualize `.ply` files

There are multiple available viewers / editors for Gaussian splatting `.ply` files.

1. [@playcanvas](https://github.com/playcanvas)'s [Super-Splat](https://github.com/playcanvas/super-splat) project ([Live demo](https://playcanvas.com/super-splat)). This is the viewer we have used for our debugging along with MeshLab.

![image](https://github.com/luciddreamer-cvlab/LucidDreamer/assets/12259041/89c4b5dd-c66f-4ad2-b1be-e5f951273049)

2. [@antimatter15](https://github.com/antimatter15)'s [WebGL viewer](https://github.com/antimatter15/splat) for Gaussian splatting ([Live demo](https://antimatter15.com/splat/)).

3. [@splinetool](https://github.com/splinetool)'s [web-based viewer](https://spline.design/) for Gaussian splatting. This is the version we have used in our project page's demo.

## 🚩 **Updates**

- ✅ December 12, 2023: We have precompiled wheels for the CUDA-based submodules and put them in `submodules/wheels`. The Windows installation guide is revised accordingly!
- ✅ December 11, 2023: We have updated installation guides for Windows. Thank you [@Maoku](https://twitter.com/Maoku) for your great contribution!
- ✅ December 8, 2023: [HuggingFace Space demo](https://huggingface.co/spaces/ironjr/LucidDreamer) is out. We deeply thank all the HF team for their support!
- ✅ December 7, 2023: [Colab](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb) implementation is now available thanks to [@camenduru](https://github.com/camenduru)!
- ✅ December 6, 2023: Code release!
- ✅ November 22, 2023: We have released our paper, LucidDreamer on [arXiv](https://arxiv.org/abs/2311.13384).

## 🌏 Citation

Please cite us if you find our project useful!

```latex
@article{chung2023luciddreamer,
    title={LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes},
    author={Chung, Jaeyoung and Lee, Suyoung and Nam, Hyeongjin and Lee, Jaerin and Lee, Kyoung Mu},
    journal={arXiv preprint arXiv:2311.13384},
    year={2023}
}
```

## 🤗 Acknowledgement

We deeply appreciate [ZoeDepth](https://github.com/isl-org/ZoeDepth), [Stability AI](), and [Runway](https://huggingface.co/runwayml/stable-diffusion-v1-5) for their models.

## 📧 Contact

If you have any questions, please email `robot0321@snu.ac.kr`, `esw0116@snu.ac.kr`, `jarin.lee@gmail.com`.

## ⭐ Star History

<a href="https://star-history.com/#luciddreamer-cvlab/LucidDreamer&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=luciddreamer-cvlab/LucidDreamer&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=luciddreamer-cvlab/LucidDreamer&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=luciddreamer-cvlab/LucidDreamer&type=Date" />
  </picture>
</a>
