import torch
import torch.nn as nn 
import dist
import numpy as np
import json
import os
import utility as ut
from tqdm.auto import tqdm
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd
import time

class MLP(nn.Module):
    def __init__(self, n_layers, n_nodes, in_features, out_fetaures, activation=torch.relu):
        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes 
        self.hidden = nn.ModuleList([nn.Flatten()])
        self.activation = activation
        for i in range(n_layers):
            if i==0:
                in_f = in_features
                if n_layers == 1:
                    out_f = out_fetaures
                else:
                    out_f = n_nodes
            else:
                in_f = n_nodes
                if i == n_layers - 1:
                    out_f = out_fetaures
                else:
                    out_f = n_nodes
            self.hidden.append(nn.Linear(in_f, out_f))
    
    def forward(self, x):
        y = x + 0
        for layer in self.hidden:
            y = self.activation(layer(y))
        return y 
    
    def save(self, folder, name):
        torch.save(self, f'{folder}/{name}')

    def load(self, folder, name):
        torch.load(self, f'{folder}/{name}')

    def re_init(self, layer_indices, distribution, alpha=None):
        """
        Re-initialize the given layers of the network.

        Parameters
        ----------
        layer_indices : int or list of int
            The indices of the layers to re-initialize.
        distribution : str
            The name of the distribution to use for re-initialization.
        alpha : float, optional
            The scale of the re-initialization. Defaults to None.
        """
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        for layer_index in layer_indices:
            out_f = self.hidden[layer_index].out_features
            tensor = getattr(dist, distribution)(self.hidden[layer_index])
            if alpha is not None:
                tensor = alpha * tensor / torch.linalg.norm(tensor) 
            self.hidden[layer_index].weight = nn.Parameter(tensor) 
            self.hidden[layer_index].bias = nn.Parameter(torch.zeros(out_f))

    def learn(self, problem, train, test, loss_function, save_folder='.', device='cpu', batch_size=64, learning_rate=1e-3, weight_decay=1., optimization_steps=100000, log_freq=100):
        """
        Learn the parameters of a model on a given problem.

        Parameters
        ----------
        problem : Problem
            The problem to learn on.
        train : torch.utils.data.Dataset
            The dataset to learn on.
        test : torch.utils.data.Dataset
            The dataset to test on.
        loss_function : str
            The loss function to use. Must be one of CrossEntropy or MSE.
        save_folder : str
            The folder to save the results in.
        device : str
            The device to use for training. Must be one of cpu or cuda.
        batch_size : int
            The batch size to use for training.
        learning_rate : float
            The learning rate to use for training.
        weight_decay : float
            The weight decay to use for training.
        optimization_steps : int
            The number of optimization steps to take.
        log_freq : int
            The frequency with which to log the loss and accuracy.

        Notes
        -----
        The loss, accuracy, and weight norm are logged at each step. The model is saved at the end of training.
        """
        loc = locals().copy()
        loc.pop('self')
        loc.pop('problem')
        loc.pop('train')
        loc.pop('test') 
        loc.pop('device')
        print(loc)

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        norms = []
        last_layer_norms = []
        log_steps = []
        t = []

        

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        log_file = f'{save_folder}/log.csv'
        if os.path.exists(log_file):
            os.remove(log_file)
        columns = ['step', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'weight_norm', 'last_layer_norm', 'time']


        steps = 0
        one_hots = torch.eye(10, 10).to(device)
        loss_fn = problem.loss_function_dict[loss_function]()

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        
        start_time = time.time()
        with tqdm(total=optimization_steps) as pbar:
            for x, labels in islice(ut.cycle(train_loader), optimization_steps):
                if (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0:
                    # Log the loss, accuracy, and weight norm at each step
                    train_losses.append(problem.compute_loss(self, train, loss_function, device, N=len(train)))
                    train_accuracies.append(problem.compute_accuracy(self, train, device, N=len(train)))
                    test_losses.append(problem.compute_loss(self, test, loss_function, device, N=len(test)))
                    test_accuracies.append(problem.compute_accuracy(self, test, device, N=len(test)))
                    log_steps.append(steps)
                    with torch.no_grad():
                        total = sum(torch.pow(p, 2).sum() for p in self.parameters())
                        norms.append(float(np.sqrt(total.item())))
                        last_layer = sum(torch.pow(p, 2).sum() for p in self.hidden[-1].parameters())
                        last_layer_norms.append(float(np.sqrt(last_layer.item())))
                    pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1] * 100, 
                        test_accuracies[-1] * 100))
                    end_time = time.time()
                    t.append(end_time - start_time)
                    start_time = end_time + 0

                optimizer.zero_grad()
                y = self(x.to(device))
                if loss_function == 'CrossEntropy':
                    loss = loss_fn(y, labels.to(device))
                elif loss_function == 'MSE':
                    loss = loss_fn(y, one_hots[labels])
                loss.backward()
                optimizer.step()
                steps += 1
                pbar.update(1)

        results = [[*elem] for elem in zip(log_steps, train_losses, test_losses, train_accuracies, test_accuracies,\
                     norms, last_layer_norms, t)]
        pd.DataFrame(results, columns=columns, dtype=float)\
            .to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))
        self.save(save_folder, 'model')
        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(log_steps, train_accuracies, color='red', label='train')
        ax.plot(log_steps, test_accuracies, color='green', label='test')
        ax.set_xscale('log')
        ax.set_xlim(10, None)
        ax.set_xlabel("Optimization Steps")
        ax.set_ylabel("Accuracy")
        ax.legend(loc=(0.015, 0.75))

        ax2 = ax.twinx()
        ax2.set_ylabel("Weight Norm", color='purple')
        ax2.plot(log_steps, last_layer_norms, color='purple', label='weight norm')
        
        fig.tight_layout()
        plt.savefig(f'{save_folder}/train-log.png', bbox_inches='tight', dpi=300)
        





