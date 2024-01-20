## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Semantic-SAM: follow [Semantic-SAM installation instructions](https://github.com/UX-Decoder/Semantic-SAM).
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name matcher python=3.8.5
conda activate matcher

pip install torch==1.13.1 torchvision==0.14.1
# or install xformers for faster inference of DINOv2
# pip install xformers==0.0.16 torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

git clone https://github.com/aim-uofa/Matcher.git
cd Matcher
pip install -r requirements.txt
```



