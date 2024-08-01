## muon-g-2-GNN
Each of the steps in the pipeline can be executed seperated followed by one of the stages.
<img width="448" alt="image" src="https://github.com/user-attachments/assets/f0579944-cc07-4163-b913-891fbac40bfe">


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
'''nvcc --version'''
If the command returns an error, it indicates that GPU is not enabled, and you should set the environment variable 'CUDA' to 'cpu'. Alternatively, if you receive a CUDA version (e.g., 9.2, 10.1, or 10.2), set the CUDA variable to cuXXX, where XXX corresponds to your CUDA version (e.g., cu92, cu101, cu102).
'''pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://pytorch-geometric.com/whl/torch-1.5.0.html'''
To ensure that all dependencies and visualization scripts are available, run the setup using:
'''pip install -e''' .
If you have access to CUDA and want to use the accelerated versions of the pipeline, you'll also need to install CuPy:
pip install cupy-cudaXXX
Replace XXX with the appropriate CUDA version suffix, such as 100, 101, 102, etc. By following these installation steps and setting the CUDA variable correctly, you can take advantage of the available resources for optimized computations with this library.


## Scripts

### `train_embed.py`
This script is used for training the muon g-2 data embeddings. It takes as input the raw data and outputs the learned embeddings. The trained embeddings can be used for further analysis or visualization.


### `convert_point_clouds.py`
This script is used to convert the data into point clouds. It takes the raw data as input and generates the embedded space. The selection of a particular query point and the query for its nearest neighbour points are done using the epsilon query ball. Make sure to import all the libraries required. The graphs with directed edges between the nodes are plotted.


### `edge_refinement.py`
This script performs edge refinement on the graphs generated from the previous step. It remove all the nodes which are not part of same track of query point from the graph.


## Data Format
The data for the training scripts should be provided in a CSV file. The CSV file should contain the necessary features and labels for the muon g-2 analysis. One dataset is given here.


## Contributions
Contributions are welcome! If you find a bug or have an improvement idea, feel free to open an issue or submit a pull request.







   



