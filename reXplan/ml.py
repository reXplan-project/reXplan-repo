import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

class NeuNet(object):       # to the class we shall provide a model, a loss_fn and an optimizer.
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class
        
        # We start by storing the arguments as attributes to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)                                                  # here we send the model to the device

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # Creates the train_step function for our model, 
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):                                                           # this is the function sending the model to the device
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):                           # data loaders provide the input data in a sutiable format to the model, in a minibatch size
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()                                                      # the model has a different behaviour during training and evaluation mode

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # Step 3 - Computes gradients for both "a" and "b" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()                                              # avoid cumulation of gradients

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()                                                       # here we set the model to evaluation mode

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # There is no need to compute Steps 3 and 4, 
            # since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn
            
    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None
            
        # Once the data loader and step function, this is the 
        # same mini-batch loop we had before
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, seed=42):                                             # this function execute the training of the model
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in tqdm(range(n_epochs)):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # If a SummaryWriter has been set...
            if self.writer:                                                         # this is optional, i.e. Tensorboard output
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        if self.writer:
            # Closes the writer
            self.writer.close()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training   

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval() 
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))                                 # sending input to device
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()                                          # sending back to cpu for return

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b', lw=1)
        plt.plot(self.val_losses, label='Test Loss', c='r', lw=1)
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        # Fetches a single mini-batch so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

PATH_MONTECARLO = r"..\jupyter_notebooks\montecarlo_database.csv" 
df_montecarlo = pd.read_csv(PATH_MONTECARLO, sep=",", index_col=[0, 1, 2, 3, 4])# , decimal=",")
column_mapping = {col: i+1 for i, col in enumerate(df_montecarlo.columns)}
timestep_mapping_df = pd.DataFrame(list(column_mapping.items()), columns=['time', 'timestep'])

PATH_NETWORK = r"..\jupyter_notebooks\network.xlsx" 
df_network = pd.read_excel(PATH_NETWORK, sheet_name="profiles", decimal=",")
df_network = df_network.drop(index=0).reset_index(drop=True)
df_network = df_network.drop(df_network.columns[0], axis=1)
temp_df = df_network.copy()
arr = temp_df.to_numpy()
arr = arr.astype(np.float64)
df_network = pd.DataFrame(np.tile(arr, (240, 1)), columns = temp_df.columns)

X_df_montecarlo = df_montecarlo.stack().unstack("id")
df_network.index= X_df_montecarlo.index
number_of_lines = len(set(X_df_montecarlo.columns))
X_df_montecarlo = pd.concat([X_df_montecarlo, df_network], axis=1)
X_df_montecarlo.insert(0, 'idx', range(1, len(X_df_montecarlo) + 1))
X = X_df_montecarlo.to_numpy()

z = pd.DataFrame(X).iloc[:,1:number_of_lines+1].astype(int).astype(str).agg(''.join, axis=1)
l = []
from collections import Counter
c = Counter(z)
for k in z:
    if c[k] == 1:
        l.append(str('G0'))
    else:
        l.append(str(k))

X_train, X_val = train_test_split(X, train_size = 0.1, stratify = pd.DataFrame(l)[0], shuffle = True, random_state = 42)

def get_opf_list():
    idx_for_opf = pd.Series(X_train[:, 0])  # Assuming X_train is a NumPy array. If not, the original code works.
    opfs_interval = (
        X_df_montecarlo[X_df_montecarlo['idx'].isin(idx_for_opf)]
        .reset_index()
        .rename(columns={'level_4': 'time'})
        [['iteration', 'time']]
        .merge(timestep_mapping_df, on='time', how='left')
    )

    grouped_list = (
        opfs_interval.groupby('iteration')['timestep']
        .apply(list)
        .reset_index()
    )
    #grouped_list.loc[grouped_list['iteration'] == 0, 'timestep'] = pd.Series([[2, 4, 50]])
    iteration_to_timestep = {row['iteration']: row['timestep'] for _, row in grouped_list.iterrows()}
    return iteration_to_timestep # grouped_list

# PATH_ENGINE = r"..\jupyter_notebooks\file\output\SimBench\engine_database.csv"
# df_engine = pd.read_csv(PATH_ENGINE, sep=",", index_col=[0, 1, 2, 3, 4])