# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn as nn
import time
import os
import logging
import ast

from sklearn.neighbors import KDTree

from torch.utils.data import DataLoader
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Load the saved model

#from train_embed.py, load the model and Hit_Pair_Dataset
model_path = "model/trained_model.pt"
net = MLP(input_dim=3, nb_hidden=256, nb_layer=3,mean=[0,0,0], std=[1,1,1], emb_dim=3)  # Instantiate the model with the same architecture as used during training
net.to(DEVICE)
net.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
net.eval()

# Now, you can use the loaded model for inference or other tasks
# For example, to perform predictions on the test dataset:
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
from hit_pair_dataset import Hit_Pair_Dataset
# Example usage
data_filepath = "/kaggle/input/dataset/time_dataset - time_dataset.csv (1).csv"
nb_samples = 16218
dataset = Hit_Pair_Dataset(data_filepath, nb_samples)

# Access elements from the dataset
h_a, h_b, t, ti, mi = dataset[0]

# Now all_preds contains the predictions made by the loaded model on the test dataset
# You can use these predictions for further analysis or evaluation

# %% [code] {"jupyter":{"outputs_hidden":false}}
def evaluate(net, experiment_dir, batch_size, eval_loader, plot_name):
    nb_batch = len(eval_loader)
    nb_eval = nb_batch * batch_size
    net.eval()

    all_preds = []
    all_labels = []
    with torch.autograd.no_grad():
        epoch_score = 0
        epoch_loss = 0
        distances = []
        embedded_points = []
        time_values = []
        muonID = []
        labels = []

        logging.info("\nEvaluating {} {} samples.".format(nb_eval, plot_name))
        print_header()
        for i, (hits_a, hits_b, target, time1, muon1) in enumerate(eval_loader):
            hits_a = hits_a.to(DEVICE)
            hits_b = hits_b.to(DEVICE)
            target = target.to(DEVICE)
            time = time1.to(DEVICE)
            muon = muon1.to(DEVICE)
#             t1 = time.time()

            emb_h_a = net(hits_a)
            emb_h_b = net(hits_b)

            pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)
            true_dist = target
            loss = nn.functional.hinge_embedding_loss(pred, true_dist)

            score = score_dist_accuracy(pred, target)
            epoch_score += score * 100
            epoch_loss += loss.item()

            nb_proc = (i+1) * batch_size
            if (i+1) % (nb_batch // 4) == 0:
                print_eval_stats(nb_proc, epoch_loss / (i+1), epoch_score / (i+1))
                
#             pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)
            pred_labels = (pred >= 0.0).float().cpu().numpy()
            true_labels = target.cpu().numpy()

            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)


            # Calculate and store the pairwise distances
            distances.extend(pred.tolist())
            embedded_points.extend(emb_h_a.tolist())
            labels.extend(target.tolist())
            time_values.extend(time1.tolist())
            muonID.extend(muon1.tolist())
            
        print_eval_stats(nb_eval, epoch_loss / nb_batch, epoch_score / nb_batch)
        

    return epoch_loss / nb_batch, epoch_score / nb_batch, distances, embedded_points, labels, time_values, muonID

eval_loss, eval_score, distances, embedded_points, labels, time_values, muonID = evaluate(net, experiment_dir, batch_size, train_loader, 'Eval')

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Convert the embedded points to a NumPy array
embedded_points_e = np.array(embedded_points)
import pandas as pd

# Create the "embed_data" folder if it doesn't exist
folder_path = '/kaggle/working/embed_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Convert the lists to a pandas DataFrame
data = {
    'Embedded Points': embedded_points,
    'Time': time_values,
    'muonID': muonID,
    'Target': labels
}
df = pd.DataFrame(data)


# Define the output file path inside the "embed_data" folder
output_path = os.path.join(folder_path, 'embedded_points_table.csv')

# Save the DataFrame as a CSV file
df.to_csv(output_path, index=False)

# %% [code] {"jupyter":{"outputs_hidden":false}}

# Define the radius of the epsilon ball
epsilon = 0.001

point_index = 34

# %% [code] {"jupyter":{"outputs_hidden":false}}
#########################################
#  Search for the nearest neighbours in the epsilon ball #
#########################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from sklearn.neighbors import KDTree
import ast

data = pd.read_csv('/kaggle/working/embed_data/embedded_points_table.csv')

# Convert the 'Embedded Points' column from string to numerical arrays
embedded_points_e = np.array([ast.literal_eval(embedded_point) for embedded_point in data['Embedded Points']])

# Convert the 'Time', 'muonID', and 'Target' columns to numpy arrays if needed
time = data["Time"].values
muonID = data["muonID"].values
labels = data['Target'].values

optimal_radius = epsilon

# Construct the KD-Tree
kdtree = KDTree(embedded_points_e)

query_point = embedded_points_e[point_index]

# Query the KD-Tree to find the indices of points within the epsilon ball
distances, indices = kdtree.query([query_point], k=len(embedded_points_e), return_distance=True)

# Filter the indices based on the distance within the epsilon ball
epsilon_indices = indices[0][distances[0] <= epsilon]

nearest_neighbors = embedded_points_e[epsilon_indices]

len(epsilon_indices)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Filter the time_data based on nearest neighbors
nearest_neighbor_indices = epsilon_indices  # Use the nearest neighbor indices obtained from the KDTree query
filtered_data = data.iloc[nearest_neighbor_indices]

# Access the filtered embedded points and time
filtered_embedded_points = filtered_data["Embedded Points"].values
filtered_time = filtered_data["Time"].values
filtered_muonID = filtered_data["muonID"].values

# Convert the embedded points to a NumPy array
embedded_points_np = np.array([eval(embedded_point) for embedded_point in filtered_embedded_points])

len(epsilon_indices)

# %% [code] {"jupyter":{"outputs_hidden":false}}
##############################################################
# X-Y plot with query point    #
#############################################################

df = pd.read_csv(/kaggle/input/data-sets/RecoOutPileup_TimeMod_uniform_1_recohitfile_training_data.csv)


# Extract the x and y coordinates of the embedded points
x_coords = df['x1']
y_coords = df['y1']

# Plot the x-y plane of embedded points
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, color='blue', label='Other Hit Points')
plt.scatter(x_coords[point_index], y_coords[point_index], color='red', label='Query Point')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.title('Embedded Points in the X-Y Plane')
plt.show()

# Create the "plots" folder if it doesn't exist
output_folder = '/kaggle/working/plots'
os.makedirs(output_folder, exist_ok=True)

# Save the plot as an image in the "plots" folder
plt.savefig(os.path.join(output_folder, 'x_y_plane_plot.png'))
plt.close()

##################################
# Plot the whole embedded points #
#################################
plt.figure(figsize=(8, 8))
plt.scatter(embedded_points_e[:, 0], embedded_points_e[:, 1], color='grey', label='Other Points')
plt.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color='blue', label='Nearest Neighbors')
plt.scatter(query_point[0], query_point[1], color='red', label='Query Point')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.title('Embedded Points Scatter Plot')
plt.show()

# Save the plot as an image in the "plots" folder
plt.savefig(os.path.join(output_folder, 'embedded_points_scatter_plot.png'))
plt.close()

# %% [code] {"jupyter":{"outputs_hidden":false}}
##########################################################
# Create graph using the query point and its neighbours #
#########################################################

# Create the "plots" folder if it doesn't exist
output_folder = '/kaggle/working/plots'
os.makedirs(output_folder, exist_ok=True)

G = nx.Graph()

# Add nodes for query point and its neighbor points
G.add_node('Query Point', color='red')
for i, neighbor_point in enumerate(nearest_neighbors):
    index = indices[0][i]  # Get the corresponding index for the neighbor point
    G.add_node(f'Neighbor {index}', color='blue', index=index)

# Add edges between query point and its neighbor points
for i in range(len(nearest_neighbors)):
    index = indices[0][i]  # Get the corresponding index for the neighbor point
    G.add_edge('Query Point', f'Neighbor {index}')

# Plot the graph
pos = nx.spring_layout(G)  # Compute node positions for visualization

plt.figure(figsize=(8, 6))
node_colors = [G.nodes[n]['color'] for n in G.nodes]
node_labels = {n: G.nodes[n]['index'] if 'index' in G.nodes[n] else '' for n in G.nodes}
nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='black', width=1)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black')
plt.title('Graph with Query Point {} and Neighbor Points (direct edge)'.format(point_index))
plt.axis('off')
# Save the plot as an image in the "plots" folder
plt.savefig(os.path.join(output_folder, 'graph_direct_edges.png'))
plt.close()
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Create a list to store the rows of the desired dataset
result_data = []

# Get the query point and its embedded coordinates
query_point = embedded_points_np[point_index]
z_query = query_point[2]
t_qp = time[point_index]

    # Loop through the nearest neighbors
for i in range(len(epsilon_indices)):
   # Get the neighbor point and its embedded coordinates
    neighbor_point = embedded_points_np[i]
    z_neighbor = neighbor_point[2]
    t_nn = time[i]

    # Calculate z_diff and t_diff for each neighbor
    z_diff_sqr = ((z_query*390.27-192.95) - (z_neighbor*390.27-192.95))**2
    time_diff_sqr = (t_qp - t_nn)**2

    # Append the row to the result_data list
    result_data.append([point_index, i, muonID[point_index], muonID[i], z_query, z_neighbor, t_qp, t_nn, z_diff_sqr, time_diff_sqr])

# Create a new DataFrame with the desired columns
result_df = pd.DataFrame(result_data, columns=["query_point", "nearest_neighbors", "muonID_query", "muonID_neighbor", "z_query", "z_neighbor", "t_qp", "t_nn", "z_diff_sqr", "time_diff_sqr"])


# Save the DataFrame to a new CSV file
result_df.to_csv("/kaggle/working/embed_data/refine_dataset.csv", index=False)
result_df
