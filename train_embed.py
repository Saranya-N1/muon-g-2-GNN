{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\n# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport matplotlib.pyplot as plt\nimport torch\nimport torch\nimport torch.nn as nn\nimport time\nimport os\nimport logging\n\nfrom torch.utils.data import DataLoader\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session\n\n# %% [code]\n#Import all the libraries\n#Install trackml, run these 3 seperately\n\n!pip install trackml\npip install --user git+https://github.com/LAL/trackml-library\nimport trackml\n\n\n# %% [code]\n#defining the class\n\n\n#####################################\n#               DATASET             #\n#####################################\n\nclass Hit_Pair_Dataset(Dataset):\n    def __init__(self, data_filepath, nb_samples):\n        dataset = pd.read_csv(\"/kaggle/input/data-sets/RecoOutPileup_TimeMod_uniform_1_recohitfile_training_data.csv\")\n\n        try:\n            self.hits_a = np.array(dataset[['x1_normed', 'y1_normed', 'z1_normed']][:nb_samples], dtype=np.float32)\n            self.hits_b = np.array(dataset[['x2_normed', 'y2_normed', 'z2_normed']][:nb_samples], dtype=np.float32)\n            self.target = np.array(dataset['label'][:nb_samples], dtype=np.float32)\n            self.time1 = np.array(dataset['t1'][:nb_samples], dtype=np.float32)\n            self.muon1 = np.array(dataset['muonID1'][:nb_samples], dtype=np.float32)\n        \n        except:\n            dim = (dataset.shape[1] - 1) // 2\n            self.hits_a = dataset[:nb_samples, :dim]\n            self.hits_b = dataset[:nb_samples, dim:2 * dim]\n            self.target = dataset[:nb_samples, -1]\n            self.time1 = dataset[:nb_samples, -1]\n            self.muon1 = dataset[:nb_samples, -1]\n            \n\n    def __getitem__(self, index):\n        h_a = self.hits_a[index]\n        h_b = self.hits_b[index]\n        t = self.target[index]\n        ti = self.time1[index]\n        mi = self.muon1[index]\n        return h_a, h_b, t, ti, mi\n\n    def __len__(self):\n        return len(self.hits_a)\n\n    def get_dim(self):\n        return self.hits_a.shape[1]\n\n# %% [code]\ndataset = Hit_Pair_Dataset(\"/kaggle/input/data-sets/RecoOutPileup_TimeMod_uniform_1_recohitfile_training_data.csv\", nb_samples=16578)\n\n# %% [code]\n#MLP model to embed the hit points\n\n\nclass MLP(nn.Module):\n    def __init__(self,\n                 nb_hidden,\n                 nb_layer,\n                 input_dim,\n                 mean,\n                 std,\n                 emb_dim=3):\n        super(MLP, self).__init__()\n        layers = [nn.Linear(input_dim, nb_hidden)]\n        ln = [nn.Linear(nb_hidden, nb_hidden) for _ in range(nb_layer-1)]\n        layers.extend(ln)\n        self.layers = nn.ModuleList(layers)\n        self.emb_layer = nn.Linear(nb_hidden, emb_dim)\n        self.act = nn.ReLU()\n        self.mean = torch.FloatTensor(mean).to(torch.float)\n        self.std = torch.FloatTensor(std).to(torch.float)\n\n    def forward(self, hits):\n        hits = self.normalize(hits)\n        for l in self.layers:\n            hits = l(hits)\n            hits = self.act(hits)\n            # hits = self.dropout(hits)\n        hits = self.emb_layer(hits)\n        return hits\n\n    def normalize(self, hits):\n\n        return hits\n\nmlp = MLP(nb_hidden=128, nb_layer=3, input_dim=3, mean=[0,0,0], std=[1,1,1])\nprint(\"MLP model:\", mlp)\n\n# %% [code]\n#score calculation: correctly predicted labels/total labels\n\ndef score_dist_accuracy(pred, true):\n    pred_classes = torch.ones_like(pred)  # Initialize all predictions as 1\n    pred_classes[pred >= 0.1] = -1 \n    correct = pred_classes == true\n    nb_correct = correct.sum().item()\n    nb_total = true.size(0)\n    score = nb_correct / nb_total\n    return score\n\n# %% [code]\n####################\n#     PRINTING     #\n####################\ndef print_header():\n  '''\n  Print header before train / evaluation run.\n  '''\n  logging.info(\"         Loss  Score\")\n\ndef print_eval_stats(nb_processed, loss, score):\n  '''\n  Log stats during train, evaluation run.\n  '''\n  logging.info(\"  {:5d}: {:.3f}  {:2.2f}\".format(nb_processed, loss, score))\n    \n#spliting the dataset for training and evaluation\n\nfrom torch.utils.data import random_split\n\nimport torch\nfrom torch.utils.data import DataLoader\n\n\nfrom sklearn.model_selection import train_test_split\n\n\n# Split the dataset into training and validation subsets\ntrain_ratio = 0.8\nvalid_ratio = 0.2\ntrain_size = int(train_ratio * len(dataset))\nvalid_size = len(dataset) - train_size\n\n# train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])\n\ntrain_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)\n\n\n# Create data loaders for training and validation\nbatch_size = 256\ntrain_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\nvalid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)\n\n# %% [code]\n\n# Check if a GPU is available, otherwise use CPU\nDEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\ninput_dim = dataset.get_dim()\nnet = MLP(input_dim=3, nb_hidden=256, nb_layer=3,mean=[0,0,0], std=[1,1,1], emb_dim=3)\nnet.to(DEVICE)\n\n# %% [code]\n#defining the model for training one epoch\n\n\nmodel_dir = \"/kaggle/working/model\"\nif not os.path.exists(model_dir):\n    os.makedirs(model_dir)\n\n\ndef train_one_epoch(net, batch_size, optimizer, train_loader):\n  net.train()\n\n  nb_batch = len(train_loader)\n  nb_train = nb_batch * batch_size\n  epoch_score = 0\n  epoch_loss  = 0\n\n\n  logging.info(\"Training on {} samples\".format(nb_train))\n  print_header()\n  t0 = time.time()\n  elapsed = 0\n  for i, (hits_a, hits_b, target, time1, muon1) in enumerate(train_loader):\n    hits_a = hits_a.to(DEVICE, non_blocking=True)\n    hits_b = hits_b.to(DEVICE, non_blocking=True)\n    target = target.to(DEVICE, non_blocking=True)\n    time1 = time1.to(DEVICE, non_blocking=True)\n    muon1 = muon1.to(DEVICE, non_blocking=True)\n    '''\n    hits_a = hits_a.to(DEVICE)\n    hits_b = hits_b.to(DEVICE)\n    target = target.to(DEVICE)\n    '''\n    optimizer.zero_grad()\n\n    emb_h_a = net(hits_a)\n    emb_h_b = net(hits_b)\n\n    pred = nn.functional.pairwise_distance(emb_h_a,emb_h_b)\n    true_dist = target\n    loss = nn.functional.hinge_embedding_loss(pred,true_dist)\n    \n    torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')\n\n    loss.backward()\n    optimizer.step()\n\n    score = score_dist_accuracy(pred, target)\n    epoch_score += score * 100\n    epoch_loss  += loss.item()\n\n    nb_proc = (i+1) * batch_size\n    if (((i+0) % (nb_batch//10)) == 0):\n        print_eval_stats(nb_proc,\n                               epoch_loss/(i+1),\n                               epoch_score/(i+1))\n\n  logging.info(\"Model elapsed:  {:.2f}\".format(elapsed))\n\n  # Save the trained model\n  torch.save(net.state_dict(), os.path.join(model_dir, \"trained_model.pt\"))\n\n  return epoch_loss / nb_batch, epoch_score / nb_batch\n\n#   Save the trained model\n# torch.save(net.state_dict(), os.path.join(model_dir, \"trained_model.pt\"))\n\n\ndef evaluate(net, batch_size, valid_loader):\n    net.eval()\n\n    nb_batch = len(valid_loader)\n    nb_valid = nb_batch * batch_size\n    valid_score = 0\n    valid_loss = 0\n\n    logging.info(\"Evaluating on {} samples\".format(nb_valid))\n    print_header()\n    t0 = time.time()\n    elapsed = 0\n    with torch.no_grad():\n        for i, (hits_a, hits_b, target, time1, muon1) in enumerate(valid_loader):\n            hits_a = hits_a.to(DEVICE, non_blocking=True)\n            hits_b = hits_b.to(DEVICE, non_blocking=True)\n            target = target.to(DEVICE, non_blocking=True)\n            time1 = time1.to(DEVICE, non_blocking=True)\n            muon1 = muon1.to(DEVICE, non_blocking=True)\n\n            emb_h_a = net(hits_a)\n            emb_h_b = net(hits_b)\n\n            pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)\n            true_dist = target\n            loss = nn.functional.hinge_embedding_loss(pred, true_dist)\n\n            score = score_dist_accuracy(pred, target)\n            valid_score += score * 100\n            valid_loss += loss.item()\n\n            nb_proc = (i + 1) * batch_size\n            if (((i + 0) % (nb_batch // 10)) == 0):\n                print_eval_stats(nb_proc, valid_loss / (i + 1), valid_score / (i + 1))\n                \n            # Save the evaluated model\n            torch.save(net.state_dict(), os.path.join(model_dir, \"evaluated_model.pt\"))\n\n    logging.info(\"Model elapsed:  {:.2f}\".format(elapsed))\n\n    return valid_loss / nb_batch, valid_score / nb_batch\n#   Save the evaluated model\n\nbatch_size = 512\n\nimport torch.optim as optim\n\n\nDEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\nnet.to(DEVICE)\n\n# Define the optimizer\noptimizer = optim.Adam(net.parameters(), lr=0.0001)\n# optimizer = optim.SGD(net.parameters(), lr=0.001)\n\n\n# Train the model\ntrain_losses=[]\ntrain_scores=[]\nvalid_losses = []\nvalid_scores = []\n\nnum_epochs = 50\nfor epoch in range(num_epochs):\n    epoch_loss, epoch_score = train_one_epoch(net, batch_size, optimizer, train_loader)\n    train_losses.append(epoch_loss)\n    train_scores.append(epoch_score)\n    valid_loss, valid_score = evaluate(net, batch_size, valid_loader)\n    valid_losses.append(valid_loss)\n    valid_scores.append(valid_score)\n#     print(f\"Epoch {epoch+1}: Loss: {epoch_loss}, Score: {epoch_score}\")\n    print(f\"Epoch {epoch+1}: Train Loss: {epoch_loss}, Train Score: {epoch_score}, Valid Loss: {valid_loss}, Valid Score: {valid_score}\")\n\n# %% [code]\n# Plot train loss and valid loss in one plot\n\n# Save the figures in the \"plots\" folder\nplots_dir = \"/kaggle/working/plots\"\nif not os.path.exists(plots_dir):\n    os.makedirs(plots_dir)\n\n# Save the Train Loss vs. Valid Loss plot\nloss_plot_filename = os.path.join(plots_dir, \"train_valid_loss_plot.png\")\n\n#Plot loss\nplt.figure(figsize=(10, 6))\nplt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')\nplt.plot(range(1, num_epochs + 1), valid_losses, label='Valid Loss', marker='o')\nplt.xlabel('Epoch')\nplt.ylabel('Loss')\nplt.title('Train Loss vs. Valid Loss')\nplt.legend()\nplt.grid()\nplt.savefig(loss_plot_filename)\nplt.show()\n\n\n# Save the Train Accuracy vs. Valid Accuracy plot\naccuracy_plot_filename = os.path.join(plots_dir, \"train_valid_accuracy_plot.png\")\nplt.figure(figsize=(10, 6))\nplt.plot(range(1, num_epochs + 1), train_scores, label='Train Accuracy', marker='o')\nplt.plot(range(1, num_epochs + 1), valid_scores, label='Valid Accuracy', marker='o')\nplt.xlabel('Epoch')\nplt.ylabel('Accuracy')\nplt.title('Train Accuracy vs. Valid Accuracy')\nplt.legend()\nplt.grid()\nplt.savefig(accuracy_plot_filename)\nplt.close()  # Close the figure to free up memory\nplt.show()\n\n# %% [code]\n#defining all the necessary functions\n\ndef track_epoch_stats(epoch_nb, lrate, train_stats, val_stats, experiment_dir):\n    print(\"Epoch: {}\".format(epoch_nb))\n    print(\"Learning rate: {:.3g}\".format(lrate))\n    print(\"Train loss: {:.3f}\".format(train_stats[0]))\n    print(\"Train score: {:.2f}\".format(train_stats[1]))\n    print(\"Validation loss: {:.3f}\".format(val_stats[0]))\n    print(\"Validation score: {:.2f}\".format(val_stats[1]))\n    print()\ndef save_test_stats(experiment_dir, test_stats):\n    stats = {'loss': test_stats[0],\n             'dist_accuracy': test_stats[1]}\n    stats_file = os.path.join(experiment_dir, TEST_STATS_FILE)\n    with open(stats_file, 'w') as f:\n        yaml.dump(stats, f, default_flow_style=False)\n        \ndef print_header():\n    '''\n    Print header before train / evaluation run.\n    '''\n    print(\"         Loss  Score\")\n\ndef print_eval_stats(nb_processed, loss, score):\n    '''\n    Log stats during train, evaluation run.\n    '''\n    print(\"  {:5d}: {:.3f}  {:2.2f}\".format(nb_processed, loss, score))\n    \nfrom torch.utils.data import random_split\nnb_samples = 16578\n\ndef main(args, force=False):\n\n  experiment_dir = os.path.join(args.artifact_storage_path, 'metric_learning_emb')\n  \n  load_path = os.path.join(args.data_storage_path, 'metric_stage_1')\n    \n  # Maybe return previously trained model\n  best_net_name = os.path.join(experiment_dir, 'best_model.pkl')\n  if os.path.isfile(best_net_name) and (not force):\n    net = load_model(best_net_name)\n    if not force:\n      print(\"Best model loaded from previous run. Not forcing training.\")\n      return net\n\n  utils.initialize_experiment_if_needed(experiment_dir, evaluate_only=False)\n#   utils.initialize_logger(experiment_dir)\n\n\n  train_path = os.path.join(load_path, 'train.pickle')\n  valid_path = os.path.join(load_path, 'valid.pickle')\n  test_path  = os.path.join(load_path, 'test.pickle')\n  stats_path = os.path.join(load_path, 'stats.yml')\n\n  train_data = Hit_Pair_Dataset(train_path, 10**8)\n  valid_data = Hit_Pair_Dataset(valid_path, 10**8)\n  test_data  = Hit_Pair_Dataset(test_path, 10**8)\n\n\n  train_dataloader = DataLoader(train_dataset,\n                                batch_size=args.batch_size,\n                                shuffle=True,\n                                drop_last=True,\n                                pin_memory=True,\n                                num_workers=8)\n  valid_dataloader = DataLoader(valid_dataset,\n                                batch_size=args.batch_size,\n                                drop_last=True,\n                                pin_memory=True,\n                                num_workers=8)\n  test_dataloader  = DataLoader(test_dataset,\n                                batch_size=args.batch_size,\n                                drop_last=True,\n                                pin_memory=True,\n                                num_workers=8)\n\n  net = create_or_restore_model(\n                                    experiment_dir, \n                                    train_dataset.get_dim(),\n                                    args.nb_hidden, \n                                    args.nb_layer,\n                                    args.emb_dim,\n                                    stats_path\n                                    )\n  net.to(DEVICE)\n  if next(net.parameters()).is_cuda:\n    logging.warning(\"Working on GPU\")\n    logging.info(\"GPU type:\\n{}\".format(torch.cuda.get_device_name(0)))\n  else:\n    logging.warning(\"Working on CPU\")\n    \n\n  train(net,\n        args.lr_start,\n        args.batch_size,\n        args.max_nb_epochs,\n        experiment_dir,\n        train_dataloader,\n        valid_dataloader)\n\n  # Perform evaluation over test set\n  try:\n    net = load_best_model(experiment_dir).to(DEVICE)\n    logging.warning(\"\\nBest model loaded for evaluation on test set.\")\n  except:\n    logging.warning(\"\\nCould not load best model for test set. Using current.\")\n  test_stats = evaluate(net, experiment_dir, args.batch_size, test_dataloader, TEST_NAME)\n  utils.save_test_stats(experiment_dir, test_stats)\n  logging.info(\"Test score:  {:3.2f}\".format(test_stats[1]))\n\n  return net\n\n# %% [code]\n\ndef save_best_model(experiment_dir, net):\n#     model_dir = os.path.join(experiment_dir, \"model\")\n    if not os.path.exists(experiment_dir):\n        os.makedirs(experiment_dir)\n    model_filename = os.path.join(experiment_dir, \"trained_full_model.pt\")\n    torch.save(net.state_dict(), model_filename)\n\ndef save_epoch_model(experiment_dir, net):\n#     model_dir = os.path.join(experiment_dir, \"model\")\n    if not os.path.exists(experiment_dir):\n        os.makedirs(experiment_dir)\n    model_filename = os.path.join(experiment_dir, \"valid_epoch_model.pt\")\n    torch.save(net.state_dict(), model_filename)\n\n\ndef train(net, lr_start, batch_size, max_nb_epochs, experiment_dir, train_loader, valid_loader):\n    optimizer = torch.optim.Adamax(net.parameters(), lr=lr_start)\n    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')\n    lr_end = lr_start / 10**3\n    best_loss = 10**10\n    # Number of epochs completed tracked in case training is interrupted\n    for i in range(max_nb_epochs):\n        t0 = time.time()\n        logging.info(\"\\nEpoch {}\".format(i+1))\n        logging.info(\"Learning rate: {0:.3g}\".format(lr_start))\n\n        train_stats = train_one_epoch(net, batch_size, optimizer, train_loader)\n        val_stats = evaluate(net, experiment_dir, batch_size, valid_loader, 'Valid')\n\n        logging.info(\"Train accuracy: {:3.2f}\".format(train_stats[1]))\n        logging.info(\"Valid accuracy: {:3.2f}\".format(val_stats[1]))\n        track_epoch_stats(i, lr_start, train_stats, val_stats, experiment_dir)\n\n        scheduler.step(val_stats[0])\n        lr_start = optimizer.param_groups[0]['lr']\n\n        if val_stats[0] < best_loss:\n            logging.warning(\"Best performance on valid set.\")\n            best_loss = val_stats[0]\n            save_best_model(experiment_dir, net)\n\n        save_epoch_model(experiment_dir, net)\n\n        logging.info(\"Epoch took {} seconds.\".format(int(time.time()-t0)))\n\n        if lr_start < lr_end:\n            break\n    logging.warning(\"Training completed.\")\n\n\ndef evaluate(net, experiment_dir, batch_size, eval_loader, plot_name):\n    nb_batch = len(eval_loader)\n    nb_eval = nb_batch * batch_size\n    net.eval()\n\n    epoch_score = 0\n    epoch_loss = 0\n\n    logging.info(\"\\nEvaluating {} {} samples.\".format(nb_eval, plot_name))\n    print_header()\n    with torch.no_grad():\n        for i, (hits_a, hits_b, target, time1, muon1) in enumerate(eval_loader):\n            hits_a = hits_a.to(DEVICE)\n            hits_b = hits_b.to(DEVICE)\n            target = target.to(DEVICE)\n            time1 = time1.to(DEVICE)\n            muon1 = muon1.to(DEVICE)\n\n            emb_h_a = net(hits_a)\n            emb_h_b = net(hits_b)\n\n            pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)\n            true_dist = target\n            loss = nn.functional.hinge_embedding_loss(pred, true_dist)\n\n            score = score_dist_accuracy(pred, target)\n            epoch_score += score * 100\n            epoch_loss += loss.item()\n\n            nb_proc = (i+1) * batch_size\n            if (i+1) % (nb_batch // 4) == 0:\n                print_eval_stats(nb_proc, epoch_loss / (i+1), epoch_score / (i+1))\n\n        print_eval_stats(nb_eval, epoch_loss / nb_batch, epoch_score / nb_batch)\n\n    # Return evaluation results as a tuple\n    return epoch_loss / nb_batch, epoch_score / nb_batch\n\n\n# Call the train function and print the results\nimport os\n\n# Check if the directory already exists.\nif not os.path.exists(\"/kaggle/working/model\"):\n\n    # Create the directory.\n    os.makedirs(\"/kaggle/working/model\")\n    \nexperiment_dir = \"/kaggle/working/model\"\nlr_start = 0.001\nmax_nb_epochs = 10\ntrain(net, lr_start, batch_size, max_nb_epochs, experiment_dir, train_loader, valid_loader)\n\n# %% [code]\ndef evaluate(net, experiment_dir, batch_size, eval_loader, plot_name):\n    nb_batch = len(eval_loader)\n    nb_eval = nb_batch * batch_size\n    net.eval()\n\n    with torch.autograd.no_grad():\n        epoch_score = 0\n        epoch_loss = 0\n        distances = []\n        embedded_points = []\n        time_values = []\n        muonID = []\n        labels = []\n\n        logging.info(\"\\nEvaluating {} {} samples.\".format(nb_eval, plot_name))\n        print_header()\n        for i, (hits_a, hits_b, target, time1, muon1) in enumerate(eval_loader):\n            hits_a = hits_a.to(DEVICE)\n            hits_b = hits_b.to(DEVICE)\n            target = target.to(DEVICE)\n            time = time1.to(DEVICE)\n            muon = muon1.to(DEVICE)\n#             t1 = time.time()\n\n            emb_h_a = net(hits_a)\n            emb_h_b = net(hits_b)\n\n            pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)\n            true_dist = target\n            loss = nn.functional.hinge_embedding_loss(pred, true_dist)\n\n            score = score_dist_accuracy(pred, target)\n            epoch_score += score * 100\n            epoch_loss += loss.item()\n\n            nb_proc = (i+1) * batch_size\n            if (i+1) % (nb_batch // 4) == 0:\n                print_eval_stats(nb_proc, epoch_loss / (i+1), epoch_score / (i+1))\n\n            # Calculate and store the pairwise distances\n            distances.extend(pred.tolist())\n            embedded_points.extend(emb_h_a.tolist())\n            labels.extend(target.tolist())\n            time_values.extend(time1.tolist())\n            muonID.extend(muon1.tolist())\n            \n        print_eval_stats(nb_eval, epoch_loss / nb_batch, epoch_score / nb_batch)\n\n    return epoch_loss / nb_batch, epoch_score / nb_batch, distances, embedded_points, labels, time_values, muonID\n\neval_loss, eval_score, distances, embedded_points, labels, time_values, muonID = evaluate(net, experiment_dir, batch_size, train_loader, 'Eval')\n\n# %% [code]\n# Convert the embedded points to a NumPy array\nembedded_points_e = np.array(embedded_points)\nimport pandas as pd\n\n# Create the \"embed_data\" folder if it doesn't exist\nfolder_path = '/kaggle/working/embed_data'\nif not os.path.exists(folder_path):\n    os.makedirs(folder_path)\n\n# Convert the lists to a pandas DataFrame\ndata = {\n    'Embedded Points': embedded_points,\n    'Time': time_values,\n    'muonID': muonID,\n    'Target': labels\n}\ndf = pd.DataFrame(data)\n\n# Define the output file path inside the \"embed_data\" folder\noutput_path = os.path.join(folder_path, 'embedded_points_table.csv')\n\n# Save the DataFrame as a CSV file\ndf.to_csv(output_path, index=False)\n\n# %% [code]\n\n\n# %% [code]\n\n\n# %% [code]\n####################################################\n#   Only for reference #\n###################################################\n\n# %% [code]\n\n# Load the saved model\nmodel_path = \"model/trained_model.pt\"\nnet = MLP(input_dim=3, nb_hidden=256, nb_layer=3,mean=[0,0,0], std=[1,1,1], emb_dim=3)  # Instantiate the model with the same architecture as used during training\nnet.to(DEVICE)\nnet.load_state_dict(torch.load(model_path))\n\n# Set the model to evaluation mode\nnet.eval()\n\n# Now, you can use the loaded model for inference or other tasks\n# For example, to perform predictions on the test dataset:\ntest_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\nfrom hit_pair_dataset import Hit_Pair_Dataset\n# Example usage\ndata_filepath = \"/kaggle/input/dataset/time_dataset - time_dataset.csv (1).csv\"\nnb_samples = 16218\ndataset = Hit_Pair_Dataset(data_filepath, nb_samples)\n\n# Access elements from the dataset\nh_a, h_b, t, ti, mi = dataset[0]\n\n\nall_preds = []\nwith torch.no_grad():\n    for hits_a, hits_b, target, time1, muon1 in test_loader:\n        hits_a = hits_a.to(DEVICE)\n        hits_b = hits_b.to(DEVICE)\n\n        # Forward pass to get predictions\n        emb_h_a = net(hits_a)\n        emb_h_b = net(hits_b)\n        pred = nn.functional.pairwise_distance(emb_h_a, emb_h_b)\n\n        # Convert predictions to numpy arrays if needed\n        pred = pred.cpu().numpy()\n        all_preds.extend(pred)\n\n# Now all_preds contains the predictions made by the loaded model on the test dataset\n# You can use these predictions for further analysis or evaluation\n\n\n# %% [code]\n############################\n#   Reference training   #\n#   Run after\n###########################\n\n#defining the model for training one epoch\n\nimport os\nimport time\nimport logging\nimport numpy as np\nimport os\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader\n\nmodel_dir = \"/kaggle/working/model\"\nif not os.path.exists(model_dir):\n    os.makedirs(model_dir)\n\n\ndef train_one_epoch(net, batch_size, optimizer, train_loader):\n  net.train()\n\n  nb_batch = len(train_loader)\n  nb_train = nb_batch * batch_size\n  epoch_score = 0\n  epoch_loss  = 0\n\n\n  logging.info(\"Training on {} samples\".format(nb_train))\n  print_header()\n  t0 = time.time()\n  elapsed = 0\n  for i, (hits_a, hits_b, target, time1, muon1) in enumerate(train_loader):\n    hits_a = hits_a.to(DEVICE, non_blocking=True)\n    hits_b = hits_b.to(DEVICE, non_blocking=True)\n    target = target.to(DEVICE, non_blocking=True)\n    time1 = time1.to(DEVICE, non_blocking=True)\n    muon1 = muon1.to(DEVICE, non_blocking=True)\n    '''\n    hits_a = hits_a.to(DEVICE)\n    hits_b = hits_b.to(DEVICE)\n    target = target.to(DEVICE)\n    '''\n    optimizer.zero_grad()\n\n    emb_h_a = net(hits_a)\n    emb_h_b = net(hits_b)\n\n    pred = nn.functional.pairwise_distance(emb_h_a,emb_h_b)\n    true_dist = target\n    loss = nn.functional.hinge_embedding_loss(pred,true_dist)\n    \n    torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')\n\n    loss.backward()\n    optimizer.step()\n\n    score = score_dist_accuracy(pred, target)\n    epoch_score += score * 100\n    epoch_loss  += loss.item()\n\n    nb_proc = (i+1) * batch_size\n    if (((i+0) % (nb_batch//10)) == 0):\n        print_eval_stats(nb_proc,\n                               epoch_loss/(i+1),\n                               epoch_score/(i+1))\n\n  logging.info(\"Model elapsed:  {:.2f}\".format(elapsed))\n\n  # Save the trained model\n  torch.save(net.state_dict(), os.path.join(model_dir, \"ref_model.pt\"))\n\n  return epoch_loss / nb_batch, epoch_score / nb_batch\n\n\n\nbatch_size = 512\n\nimport torch.optim as optim\n\n\nDEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\nnet.to(DEVICE)\n\n# Define the optimizer\noptimizer = optim.Adam(net.parameters(), lr=0.0001)\n# optimizer = optim.SGD(net.parameters(), lr=0.001)\n\n\n# Train the model\ntrain_losses=[]\ntrain_scores=[]\n\nnum_epochs = 50\nfor epoch in range(num_epochs):\n    epoch_loss, epoch_score = train_one_epoch(net, batch_size, optimizer, train_loader)\n    train_losses.append(epoch_loss)\n    train_scores.append(epoch_score)\n#     print(f\"Epoch {epoch+1}: Loss: {epoch_loss}, Score: {epoch_score}\")\n    print(f\"Epoch {epoch+1}: Train Loss: {epoch_loss}, Train Score: {epoch_score}\")","metadata":{"_uuid":"47dae99c-88af-403f-9c81-61f8404f2e6b","_cell_guid":"909cd70d-77ca-4053-b0ff-4c0522cbc2cf","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}