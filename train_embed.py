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
#Import all the libraries
#Install trackml, run these 3 seperately

!pip install trackml
pip install --user git+https://github.com/LAL/trackml-library
import trackml

# %% [code] {"jupyter":{"outputs_hidden":false}}
#defining the class


#####################################
#               DATASET             #
#####################################

class Hit_Pair_Dataset(Dataset):
    def __init__(self, data_filepath, nb_samples):
        dataset = pd.read_csv("/kaggle/input/data-sets/RecoOutPileup_TimeMod_uniform_1_recohitfile_training_data.csv")

        try:
            self.hits_a = np.array(dataset[['x1_normed', 'y1_normed', 'z1_normed']][:nb_samples], dtype=np.float32)
            self.hits_b = np.array(dataset[['x2_normed', 'y2_normed', 'z2_normed']][:nb_samples], dtype=np.float32)
            self.target = np.array(dataset['label'][:nb_samples], dtype=np.float32)
            self.time1 = np.array(dataset['t1'][:nb_samples], dtype=np.float32)
            self.muon1 = np.array(dataset['muonID1'][:nb_samples], dtype=np.float32)
        
        except:
            dim = (dataset.shape[1] - 1) // 2
            self.hits_a = dataset[:nb_samples, :dim]
            self.hits_b = dataset[:nb_samples, dim:2 * dim]
            self.target = dataset[:nb_samples, -1]
            self.time1 = dataset[:nb_samples, -1]
            self.muon1 = dataset[:nb_samples, -1]
            

    def __getitem__(self, index):
        h_a = self.hits_a[index]
        h_b = self.hits_b[index]
        t = self.target[index]
        ti = self.time1[index]
        mi = self.muon1[index]
        return h_a, h_b, t, ti, mi

    def __len__(self):
        return len(self.hits_a)

    def get_dim(self):
        return self.hits_a.shape[1]

# %% [code] {"jupyter":{"outputs_hidden":false}}
dataset = Hit_Pair_Dataset("/kaggle/input/data-sets/RecoOutPileup_TimeMod_uniform_1_recohitfile_training_data.csv", nb_samples=16578)

# %% [code] {"jupyter":{"outputs_hidden":false}}
#MLP model to embed the hit points


class MLP(nn.Module):
    def __init__(self,
                 nb_hidden,
                 nb_layer,
                 input_dim,
                 mean,
                 std,
                 emb_dim=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, nb_hidden)]
        ln = [nn.Linear(nb_hidden, nb_hidden) for _ in range(nb_layer-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(nb_hidden, emb_dim)
        self.act = nn.ReLU()
        self.mean = torch.FloatTensor(mean).to(torch.float)
        self.std = torch.FloatTensor(std).to(torch.float)

    def forward(self, hits):
        hits = self.normalize(hits)
        for l in self.layers:
            hits = l(hits)
            hits = self.act(hits)
            # hits = self.dropout(hits)
        hits = self.emb_layer(hits)
        return hits

    def normalize(self, hits):

        return hits

mlp = MLP(nb_hidden=128, nb_layer=3, input_dim=3, mean=[0,0,0], std=[1,1,1])
print("MLP model:", mlp)

# %% [code] {"jupyter":{"outputs_hidden":false}}
#score calculation: correctly predicted labels/total labels

def score_dist_accuracy(pred, true):
    pred_classes = torch.ones_like(pred)  # Initialize all predictions as 1
    pred_classes[pred >= 0.1] = -1 
    correct = pred_classes == true
    nb_correct = correct.sum().item()
    nb_total = true.size(0)
    score = nb_correct / nb_total
    return score

# %% [code] {"jupyter":{"outputs_hidden":false}}
####################
#     PRINTING     #
####################
def print_header():
  '''
  Print header before train / evaluation run.
  '''
  logging.info("         Loss  Score")

def print_eval_stats(nb_processed, loss, score):
  '''
  Log stats during train, evaluation run.
  '''
  logging.info("  {:5d}: {:.3f}  {:2.2f}".format(nb_processed, loss, score))
    
#spliting the dataset for training and evaluation

from torch.utils.data import random_split

import torch
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


# Split the dataset into training and validation subsets
train_ratio = 0.8
valid_ratio = 0.2
train_size = int(train_ratio * len(dataset))
valid_size = len(dataset) - train_size

# train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


# Create data loaders for training and validation
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}

# Check if a GPU is available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = dataset.get_dim()
net = MLP(input_dim=3, nb_hidden=256, nb_layer=3,mean=[0,0,0], std=[1,1,1], emb_dim=3)
net.to(DEVICE)

# %% [code] {"jupyter":{"outputs_hidden":false}}
#defining the model for training one epoch


model_dir = "/kaggle/working/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def train_one_epoch(net, batch_size, optimizer, train_loader):
  net.train()

  nb_batch = len(train_loader)
  nb_train = nb_batch * batch_size
  epoch_score = 0
  epoch_loss  = 0


  logging.info("Training on {} samples".format(nb_train))
  print_header()
  t0 = time.time()
  elapsed = 0
  for i, (hits_a, hits_b, target, time1, muon1) in enumerate(train_loader):
    hits_a = hits_a.to(DEVICE, non_blocking=True)
    hits_b = hits_b.to(DEVICE, non_blocking=True)
    target = target.to(DEVICE, non_blocking=True)
    time1 = time1.to(DEVICE, non_blocking=True)
    muon1 = muon1.to(DEVICE, non_blocking=True)
    '''
    hits_a = hits_a.to(DEVICE)
    hits_b = hits_b.to(DEVICE)
    target = target.to(DEVICE)
    '''
    optimizer.zero_grad()

    emb_h_a = net(hits_a)
    emb_h_b = net(hits_b)

    pred = nn.functional.pairwise_distance(emb_h_a,emb_h_b)
    true_dist = target
    loss = nn.functional.hinge_embedding_loss(pred,true_dist)
    
    torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')

    loss.backward()
    optimizer.step()

    score = score_dist_accuracy(pred, target)
    epoch_score += score * 100
    epoch_loss  += loss.item()

    nb_proc = (i+1) * batch_size
    if (((i+0) % (nb_batch//10)) == 0):
        print_eval_stats(nb_proc,
                               epoch_loss/(i+1),
                               epoch_score/(i+1))

  logging.info("Model elapsed:  {:.2f}".format(elapsed))

  # Save the trained model
  torch.save(net.state_dict(), os.path.join(model_dir, "trained_model.pt"))

  return epoch_loss / nb_batch, epoch_score / nb_batch

#   Save the trained model
# torch.save(net.state_dict(), os.path.join(model_dir, "trained_model.pt"))


def evaluate(net, batch_size, valid_loader):
    net.eval()

    nb_batch = len(valid_loader)
    nb_valid = nb_batch * batch_size
    valid_score = 0
    valid_loss = 0

    logging.info("Evaluating on {} samples".format(nb_valid))
    print_header()
    t0 = time.time()
    elapsed = 0
    with torch.no_grad():
        for i, (hits_a, hits_b, target, time1, muon1) in enumerate(valid_loader):
            hits_a = hits_a.to(DEVICE, non_blocking=True)
            hits_b = hits_b.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            time1 = time1.to(DEVICE, non_blocking=True)
            muon1 = muon1.to(DEVICE, non_blocking=True)

            emb_h_a = net(hits_a)
            emb_h_b = net(hits_b)

            pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)
            true_dist = target
            loss = nn.functional.hinge_embedding_loss(pred, true_dist)

            score = score_dist_accuracy(pred, target)
            valid_score += score * 100
            valid_loss += loss.item()

            nb_proc = (i + 1) * batch_size
            if (((i + 0) % (nb_batch // 10)) == 0):
                print_eval_stats(nb_proc, valid_loss / (i + 1), valid_score / (i + 1))
                
            # Save the evaluated model
            torch.save(net.state_dict(), os.path.join(model_dir, "evaluated_model.pt"))

    logging.info("Model elapsed:  {:.2f}".format(elapsed))

    return valid_loss / nb_batch, valid_score / nb_batch
#   Save the evaluated model

batch_size = 512

import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# optimizer = optim.SGD(net.parameters(), lr=0.001)


# Train the model
train_losses=[]
train_scores=[]
valid_losses = []
valid_scores = []

num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss, epoch_score = train_one_epoch(net, batch_size, optimizer, train_loader)
    train_losses.append(epoch_loss)
    train_scores.append(epoch_score)
    valid_loss, valid_score = evaluate(net, batch_size, valid_loader)
    valid_losses.append(valid_loss)
    valid_scores.append(valid_score)
#     print(f"Epoch {epoch+1}: Loss: {epoch_loss}, Score: {epoch_score}")
    print(f"Epoch {epoch+1}: Train Loss: {epoch_loss}, Train Score: {epoch_score}, Valid Loss: {valid_loss}, Valid Score: {valid_score}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Plot train loss and valid loss in one plot

# Save the figures in the "plots" folder
plots_dir = "/kaggle/working/plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Save the Train Loss vs. Valid Loss plot
loss_plot_filename = os.path.join(plots_dir, "train_valid_loss_plot.png")

#Plot loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), valid_losses, label='Valid Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs. Valid Loss')
plt.legend()
plt.grid()
plt.savefig(loss_plot_filename)
plt.show()


# Save the Train Accuracy vs. Valid Accuracy plot
accuracy_plot_filename = os.path.join(plots_dir, "train_valid_accuracy_plot.png")
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_scores, label='Train Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), valid_scores, label='Valid Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs. Valid Accuracy')
plt.legend()
plt.grid()
plt.savefig(accuracy_plot_filename)
plt.close()  # Close the figure to free up memory
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
#defining all the necessary functions

def track_epoch_stats(epoch_nb, lrate, train_stats, val_stats, experiment_dir):
    print("Epoch: {}".format(epoch_nb))
    print("Learning rate: {:.3g}".format(lrate))
    print("Train loss: {:.3f}".format(train_stats[0]))
    print("Train score: {:.2f}".format(train_stats[1]))
    print("Validation loss: {:.3f}".format(val_stats[0]))
    print("Validation score: {:.2f}".format(val_stats[1]))
    print()
def save_test_stats(experiment_dir, test_stats):
    stats = {'loss': test_stats[0],
             'dist_accuracy': test_stats[1]}
    stats_file = os.path.join(experiment_dir, TEST_STATS_FILE)
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
        
def print_header():
    '''
    Print header before train / evaluation run.
    '''
    print("         Loss  Score")

def print_eval_stats(nb_processed, loss, score):
    '''
    Log stats during train, evaluation run.
    '''
    print("  {:5d}: {:.3f}  {:2.2f}".format(nb_processed, loss, score))
    
from torch.utils.data import random_split
nb_samples = 16578

def main(args, force=False):

  experiment_dir = os.path.join(args.artifact_storage_path, 'metric_learning_emb')
  
  load_path = os.path.join(args.data_storage_path, 'metric_stage_1')
    
  # Maybe return previously trained model
  best_net_name = os.path.join(experiment_dir, 'best_model.pkl')
  if os.path.isfile(best_net_name) and (not force):
    net = load_model(best_net_name)
    if not force:
      print("Best model loaded from previous run. Not forcing training.")
      return net

  utils.initialize_experiment_if_needed(experiment_dir, evaluate_only=False)
#   utils.initialize_logger(experiment_dir)


  train_path = os.path.join(load_path, 'train.pickle')
  valid_path = os.path.join(load_path, 'valid.pickle')
  test_path  = os.path.join(load_path, 'test.pickle')
  stats_path = os.path.join(load_path, 'stats.yml')

  train_data = Hit_Pair_Dataset(train_path, 10**8)
  valid_data = Hit_Pair_Dataset(valid_path, 10**8)
  test_data  = Hit_Pair_Dataset(test_path, 10**8)


  train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=8)
  valid_dataloader = DataLoader(valid_dataset,
                                batch_size=args.batch_size,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=8)
  test_dataloader  = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=8)

  net = create_or_restore_model(
                                    experiment_dir, 
                                    train_dataset.get_dim(),
                                    args.nb_hidden, 
                                    args.nb_layer,
                                    args.emb_dim,
                                    stats_path
                                    )
  net.to(DEVICE)
  if next(net.parameters()).is_cuda:
    logging.warning("Working on GPU")
    logging.info("GPU type:\n{}".format(torch.cuda.get_device_name(0)))
  else:
    logging.warning("Working on CPU")
    

  train(net,
        args.lr_start,
        args.batch_size,
        args.max_nb_epochs,
        experiment_dir,
        train_dataloader,
        valid_dataloader)

  # Perform evaluation over test set
  try:
    net = load_best_model(experiment_dir).to(DEVICE)
    logging.warning("\nBest model loaded for evaluation on test set.")
  except:
    logging.warning("\nCould not load best model for test set. Using current.")
  test_stats = evaluate(net, experiment_dir, args.batch_size, test_dataloader, TEST_NAME)
  utils.save_test_stats(experiment_dir, test_stats)
  logging.info("Test score:  {:3.2f}".format(test_stats[1]))

  return net

# %% [code] {"jupyter":{"outputs_hidden":false}}

def save_best_model(experiment_dir, net):
#     model_dir = os.path.join(experiment_dir, "model")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    model_filename = os.path.join(experiment_dir, "trained_full_model.pt")
    torch.save(net.state_dict(), model_filename)

def save_epoch_model(experiment_dir, net):
#     model_dir = os.path.join(experiment_dir, "model")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    model_filename = os.path.join(experiment_dir, "valid_epoch_model.pt")
    torch.save(net.state_dict(), model_filename)


def train(net, lr_start, batch_size, max_nb_epochs, experiment_dir, train_loader, valid_loader):
    optimizer = torch.optim.Adamax(net.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_end = lr_start / 10**3
    best_loss = 10**10
    # Number of epochs completed tracked in case training is interrupted
    for i in range(max_nb_epochs):
        t0 = time.time()
        logging.info("\nEpoch {}".format(i+1))
        logging.info("Learning rate: {0:.3g}".format(lr_start))

        train_stats = train_one_epoch(net, batch_size, optimizer, train_loader)
        val_stats = evaluate(net, experiment_dir, batch_size, valid_loader, 'Valid')

        logging.info("Train accuracy: {:3.2f}".format(train_stats[1]))
        logging.info("Valid accuracy: {:3.2f}".format(val_stats[1]))
        track_epoch_stats(i, lr_start, train_stats, val_stats, experiment_dir)

        scheduler.step(val_stats[0])
        lr_start = optimizer.param_groups[0]['lr']

        if val_stats[0] < best_loss:
            logging.warning("Best performance on valid set.")
            best_loss = val_stats[0]
            save_best_model(experiment_dir, net)

        save_epoch_model(experiment_dir, net)

        logging.info("Epoch took {} seconds.".format(int(time.time()-t0)))

        if lr_start < lr_end:
            break
    logging.warning("Training completed.")


def evaluate(net, experiment_dir, batch_size, eval_loader, plot_name):
    nb_batch = len(eval_loader)
    nb_eval = nb_batch * batch_size
    net.eval()

    epoch_score = 0
    epoch_loss = 0

    logging.info("\nEvaluating {} {} samples.".format(nb_eval, plot_name))
    print_header()
    with torch.no_grad():
        for i, (hits_a, hits_b, target, time1, muon1) in enumerate(eval_loader):
            hits_a = hits_a.to(DEVICE)
            hits_b = hits_b.to(DEVICE)
            target = target.to(DEVICE)
            time1 = time1.to(DEVICE)
            muon1 = muon1.to(DEVICE)

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

        print_eval_stats(nb_eval, epoch_loss / nb_batch, epoch_score / nb_batch)

    # Return evaluation results as a tuple
    return epoch_loss / nb_batch, epoch_score / nb_batch


# Call the train function and print the results
import os

# Check if the directory already exists.
if not os.path.exists("/kaggle/working/model"):

    # Create the directory.
    os.makedirs("/kaggle/working/model")
    
experiment_dir = "/kaggle/working/model"
lr_start = 0.001
max_nb_epochs = 10
train(net, lr_start, batch_size, max_nb_epochs, experiment_dir, train_loader, valid_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def evaluate(net, experiment_dir, batch_size, eval_loader, plot_name):
    nb_batch = len(eval_loader)
    nb_eval = nb_batch * batch_size
    net.eval()

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


# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
####################################################
#   Only for reference #
###################################################

# %% [code] {"jupyter":{"outputs_hidden":false}}

# Load the saved model
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


all_preds = []
with torch.no_grad():
    for hits_a, hits_b, target, time1, muon1 in test_loader:
        hits_a = hits_a.to(DEVICE)
        hits_b = hits_b.to(DEVICE)

        # Forward pass to get predictions
        emb_h_a = net(hits_a)
        emb_h_b = net(hits_b)
        pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)

        # Convert predictions to numpy arrays if needed
        pred = pred.cpu().numpy()
        all_preds.extend(pred)

# Now all_preds contains the predictions made by the loaded model on the test dataset
# You can use these predictions for further analysis or evaluation

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################
#   Reference training  #
###########################

#defining the model for training one epoch

import os
import time
import logging
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model_dir = "/kaggle/working/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def train_one_epoch(net, batch_size, optimizer, train_loader):
  net.train()

  nb_batch = len(train_loader)
  nb_train = nb_batch * batch_size
  epoch_score = 0
  epoch_loss  = 0


  logging.info("Training on {} samples".format(nb_train))
  print_header()
  t0 = time.time()
  elapsed = 0
  for i, (hits_a, hits_b, target, time1, muon1) in enumerate(train_loader):
    hits_a = hits_a.to(DEVICE, non_blocking=True)
    hits_b = hits_b.to(DEVICE, non_blocking=True)
    target = target.to(DEVICE, non_blocking=True)
    time1 = time1.to(DEVICE, non_blocking=True)
    muon1 = muon1.to(DEVICE, non_blocking=True)
    '''
    hits_a = hits_a.to(DEVICE)
    hits_b = hits_b.to(DEVICE)
    target = target.to(DEVICE)
    '''
    optimizer.zero_grad()

    emb_h_a = net(hits_a)
    emb_h_b = net(hits_b)

    pred = nn.functional.pairwise_distance(emb_h_a,emb_h_b)
    true_dist = target
    loss = nn.functional.hinge_embedding_loss(pred,true_dist)
    
    torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')

    loss.backward()
    optimizer.step()

    score = score_dist_accuracy(pred, target)
    epoch_score += score * 100
    epoch_loss  += loss.item()

    nb_proc = (i+1) * batch_size
    if (((i+0) % (nb_batch//10)) == 0):
        print_eval_stats(nb_proc,
                               epoch_loss/(i+1),
                               epoch_score/(i+1))

  logging.info("Model elapsed:  {:.2f}".format(elapsed))

  # Save the trained model
  torch.save(net.state_dict(), os.path.join(model_dir, "ref_model.pt"))

  return epoch_loss / nb_batch, epoch_score / nb_batch



batch_size = 512

import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# optimizer = optim.SGD(net.parameters(), lr=0.001)


# Train the model
train_losses=[]
train_scores=[]

num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss, epoch_score = train_one_epoch(net, batch_size, optimizer, train_loader)
    train_losses.append(epoch_loss)
    train_scores.append(epoch_score)
#     print(f"Epoch {epoch+1}: Loss: {epoch_loss}, Score: {epoch_score}")
    print(f"Epoch {epoch+1}: Train Loss: {epoch_loss}, Train Score: {epoch_score}")
