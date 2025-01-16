# Diffusion and Stable Diffusion Training and Generation

This project implements diffusion models for image generation with conditional context. Discusses diffusion and stable diffusion
It includes two main notebooks:

1. `Diffusion.ipynb`: Core implementation of the diffusion model training and sampling
2. `Stable_Diffusion.ipynb`: Implementation using the Stable Diffusion model for image generation with custom loss functions

## Setup

1. Install dependencies:
```bash
pip install -r src/requirements.txt
```

2. Required dependencies:
- PyTorch
- torchvision 
- Pillow
- numpy
- pandas
- transformers
- diffusers
- scipy
- accelerate
- tqdm
- matplotlib
- IPython
- ftfy
- opencv-python

## Project Structure

- `Diffusion.ipynb`: Main notebook for training the diffusion model
- `Stable_Diffusion.ipynb`: Notebook for Stable Diffusion experiments
- `diffusion_utilities.py`: Helper functions and model components
- `requirements.txt`: List of Python package dependencies

## Features

### Diffusion Model (`Diffusion.ipynb`)
- Implements denoising diffusion probabilistic models
- Includes conditional generation with context vectors
- Uses U-Net architecture with residual blocks
- Supports training on custom datasets

### Stable Diffusion (`Stable_Diffusion.ipynb`) 
- Uses pre-trained Stable Diffusion model
- Implements custom loss functions:
  - Blue loss
  - Elastic loss
  - Symmetry loss
  - Saturation loss
- Supports image generation with loss-based guidance

## Usage

1. Training the diffusion model:
- Open `src/Diffusion.ipynb`
- Configure hyperparameters as needed
- Run cells to train model and generate samples

2. Using Stable Diffusion:
- Open `src/Stable_Diffusion.ipynb`
- Set up desired prompts and loss functions
- Run cells to generate images with custom guidance

## Acknowledgments

- Sprites by ElvGames, [FrootsnVeggies](https://zrghr.itch.io/froots-and-veggies-culinary-pixels) and [kyrise](https://kyrise.itch.io/)
- Code modified from [minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- Based on papers:
  - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  - [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)


2. Run the notebooks:
```bash
jupyter notebook src/Diffusion.ipynb
jupyter notebook src/Stable_Diffusion.ipynb
```
