# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false}}


df = pd.read_csv('/kaggle/working/embed_data/refine_dataset.csv')
def sigmoid(x):
 return 1/(1 + np.exp(-x))

z_diff_sqr = df['z_diff_sqr']
t_diff_sqr = df['time_diff_sqr']

def spatial_temporal_sigmoid(z_diff_sqr,time_diff_sqr,spatial_weight, temporal_weight):
  """Returns a sigmoid function value that is high if two hits are spatially and temporally close and low otherwise."""

  spatial_term = sigmoid(spatial_weight / z_diff_sqr)
  temporal_term = sigmoid(temporal_weight / (time_diff_sqr+.00001))
#   if (z_diff_sqr [0]< 0.0001):
#     return 0
  return spatial_term * temporal_term

df["prob"] = spatial_temporal_sigmoid(df["z_diff_sqr"],df["time_diff_sqr"],100,1)
# df['class_label'] = df.apply(class_label, axis=1)
df.to_csv("/kaggle/working/embed_data/dataset_refinement_test.csv", index=False)
df

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Create the "plots" folder if it doesn't exist
output_folder = '/kaggle/working/plots'
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
df = pd.read_csv('/kaggle/working/dataset_refinement_test1.csv')

# Create a graph with all edges between query point and nearest neighbors
graph_all_edges = nx.Graph()
for _, row in df.iterrows():
    graph_all_edges.add_edge("Query", row['nearest_neighbors'])

# Create a graph with edges only between query point and nearest neighbors with probability > 0.5
graph_filtered_edges = nx.Graph()
for _, row in df[df['prob'] > 0.51].iterrows():
    graph_filtered_edges.add_edge("Query", row['nearest_neighbors'])

# Set the positions of the nodes for visualization
pos = nx.spring_layout(graph_all_edges)

# Extract muonID_neighbor from the DataFrame
muonID_dict = dict(df[['nearest_neighbors', 'muonID_neighbor']].values)

# Create a color map for node colors based on muonID_neighbor
unique_muonID_neighbors = df['muonID_neighbor'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_muonID_neighbors)))
muonID_color_map = dict(zip(unique_muonID_neighbors, colors))

# Draw the first graph with all edges
plt.figure(figsize=(10, 8))
node_colors_all_edges = [muonID_color_map[muonID_dict[node]] if node in muonID_dict else 'red' for node in graph_all_edges.nodes]
nx.draw(graph_all_edges, pos, with_labels=True, labels=muonID_dict, node_color=node_colors_all_edges, node_size=200, font_size=10, font_weight='bold', font_color = 'white')
nx.draw_networkx_nodes(graph_all_edges, pos, nodelist=['Query'], node_color='red', node_size=2000)
nx.draw_networkx_nodes(graph_all_edges, pos, nodelist=graph_all_edges.nodes()-{'Query'}, node_color=[muonID_color_map[muonID_dict[node]] for node in graph_all_edges.nodes()-{'Query'}], node_size=800)
plt.title("Graph with All Edges")
plt.savefig(os.path.join(output_folder, 'graph_direct_edges_muon_details.png'))
plt.close()
plt.show()

# Draw the second graph with filtered edges
plt.figure(figsize=(10, 8))
node_colors_filtered_edges = [muonID_color_map[muonID_dict[node]] if node in muonID_dict else 'red' for node in graph_filtered_edges.nodes]
nx.draw(graph_filtered_edges, pos, with_labels=True, labels=muonID_dict, node_color=node_colors_filtered_edges, node_size=200, font_size=10, font_weight='bold', font_color = 'white')
nx.draw_networkx_nodes(graph_filtered_edges, pos, nodelist=['Query'], node_color='red', node_size=2000)
nx.draw_networkx_nodes(graph_filtered_edges, pos, nodelist=graph_filtered_edges.nodes()-{'Query'}, node_color=[muonID_color_map[muonID_dict[node]] for node in graph_filtered_edges.nodes()-{'Query'}], node_size=800)
plt.title("Graph with Edges (prob > 0.5)")
plt.savefig(os.path.join(output_folder, 'graph_refined.png'))
plt.close()
plt.show()

len(num_edges_filtered)

# %% [code] {"jupyter":{"outputs_hidden":false}}


# Load the dataset
df = pd.read_csv('/kaggle/working/dataset_refinement_test1.csv')

# Create a graph with edges only between query point and nearest neighbors with probability > 0.5
graph_filtered_edges = nx.Graph()
for _, row in df[df['prob'] > 0.51].iterrows():
    graph_filtered_edges.add_edge("Query", row['nearest_neighbors'])

# Count the number of edges in the filtered graph
num_edges_filtered = graph_filtered_edges.number_of_edges()

# Print the number of edges in the filtered graph
print("Number of edges in the filtered graph:", num_edges_filtered)

# Load the other dataset (replace 'other_dataset.csv' with the actual filename)
other_df = pd.read_csv('/kaggle/input/data-sets/RecoOutPileup_TimeMod_uniform_1_recohitfile_training_data.csv')


# Step 2: Calculate the number of rows in the refined dataset where muonID_query is equal to muonID_neighbor
num_rows_refined_dataset = df[df['muonID_query'] == df['muonID_neighbor']].shape[0]

# Step 3: Calculate the number of rows in the real dataset where muonID is equal to the muonID_neighbor in the filtered graph
num_rows_real_dataset = other_df[other_df['muonID1'].isin(df['muonID_query'])].shape[0]

# Step 4: Calculate the graph reconstruction efficiency
graph_reconstruction_efficiency = num_edges_filtered / num_rows_refined_dataset

# Step 5: Calculate the track reconstruction efficiency
track_reconstruction_efficiency = num_edges_filtered / num_rows_real_dataset

print("Graph Reconstruction Efficiency:", graph_reconstruction_efficiency)
print("Track Reconstruction Efficiency:", track_reconstruction_efficiency)
