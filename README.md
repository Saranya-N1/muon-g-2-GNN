## muon-g-2-GNN
Each of the steps in the pipeline can be executed seperated followed by one of the stages.
![Screenshot (360)](https://github.com/Saranya-N1/muon-g-2-GNN/assets/99310392/26a4157a-01fe-43b6-bb68-f87b525f0b79)

##Setting Up
#Environment
These codes are written in Kaggle notebook. This will work in any python platform with some changes.

## Description
This repository contains code for performing first few steps of track finding algorithm using three Python scripts: `train_embed.py`, `convert_point_clouds.py`, and `edge_refinement.py`.

## Installation
1. Clone this repository to your local machine:
git clone https://github.com/Saranya-N1/muon-g-2-GNN.git
2. Install the required dependencies: 
'''pip install --user git+https://github.com/LAL/trackml-library'''
3. To make full use of the available system resources, this library offers two versions of certain functions - one optimized for CUDA GPU acceleration and the other for CPU operations. Before installation, determine your CUDA version by running the command
# nvcc --version.
If the command returns an error, it indicates that GPU is not enabled, and you should set the environment variable 'CUDA' to 'cpu'. Alternatively, if you receive a CUDA version (e.g., 9.2, 10.1, or 10.2), set the CUDA variable to cuXXX, where XXX corresponds to your CUDA version (e.g., cu92, cu101, cu102).
pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://pytorch-geometric.com/whl/torch-1.5.0.html

   



