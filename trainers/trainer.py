import numpy as np
import torch
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

from losses.loss import DetectionLoss, VAELoss
from models.detector import Detector
from utility.plotter import plot_training_loss

class TrainerDetector:
    def __init__(self, params, detector):

        self.params = params
        self.device = params['device']

        self.detector = detector
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=params['lr_detector'])
        self.criterion = DetectionLoss()

    def train_step(self, x, y):
        self.optimizer.zero_grad()

        y_pred = self.detector(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_loader):
        train_losses, val_losses = [], []

        start_training = time.time()

        try:
            for epoch in tqdm(range(self.params['epochs_detector']), total=self.params['epochs_detector']):
                self.detector.train()
                loss, count = 0, 0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    batch_loss = self.train_step(x, y)

                    loss += batch_loss.item()

                loss /= len(train_loader)
                train_losses.append(loss)

                print(f'Epoch: {epoch}, Loss: {loss} \n')

                with open(self.params['log_path'], 'a') as f:
                    f.write(f'Epoch: {epoch}, Loss: {loss} \n')

        except KeyboardInterrupt:
            print('Training interrupted.')

        end_training = time.time()
        elapsed_time = end_training - start_training

        torch.save(self.detector.state_dict(), self.params['last_detector_checkpoint_path'])

        print(f'Elapsed time for training: {elapsed_time} \n')
        with open(self.params['log_path'], 'a') as f:
            f.write(f'Elapsed time for training: {elapsed_time} \n')

        plot_training_loss(train_losses, self.params)

    def evaluate(self, val_loader):
        val_losses = 0
        self.detector.eval()

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.detector(x)
                loss = self.criterion(y_pred, y)
                val_losses += loss

        return val_losses / len(val_loader)

    def test(self, test_loader, detector):
        detector.eval()
        count = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)

                y_hat = detector(x)

                if count == 0:
                    y_true, y_pred = y, y_hat.detach().cpu()
                else:
                    y_true = torch.cat((y_true, y), dim=0)
                    y_pred = torch.cat((y_pred, y_hat.detach().cpu()), dim=0)

                count += 1

        return y_pred, y_true

class TrainerVAE:
    def __init__(self, params, vae):
        self.params = params
        self.device = params['device']
        self.vae = vae

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.params['lr_vae'])
        self.criterion = VAELoss()

    def train(self, train_loader):
        train_losses = []
        train_mse_losses, train_kld_losses = [], []

        start_training = time.time()

        try:
            for epoch in tqdm(range(self.params['epochs_vae']), total=self.params['epochs_vae']):
                self.vae.train()
                loss = 0
                count = 0
                mse_loss = 0
                kld_loss = 0

                for x, _ in train_loader:
                    x = x.to(self.device)
                    self.optimizer.zero_grad()

                    x_rec, mu, log_var = self.vae(x)
                    loss_batch, mse_loss_batch, kld_loss_batch = self.criterion(x_rec, x,  mu, log_var)
                    loss_batch.backward()
                    self.optimizer.step()

                    loss += loss_batch.item()
                    mse_loss += mse_loss_batch.item()
                    kld_loss += kld_loss_batch.item()

                loss /= len(train_loader)
                mse_loss /= len(train_loader)
                kld_loss /= len(train_loader)
                train_losses.append(loss)
                train_mse_losses.append(mse_loss)
                train_kld_losses.append(kld_loss)

                print(f'\n Epoch: {epoch}, Loss: {loss} (MSE: {mse_loss}, KLD: {kld_loss})  \n')

                with open(self.params['log_path'], 'a') as f:
                    f.write(f'\n Epoch: {epoch}, Loss: {loss} (MSE: {mse_loss}, KLD: {kld_loss}) \n')

        except KeyboardInterrupt:
            print('Training interrupted.')

        end_training = time.time()
        elapsed_time = end_training - start_training

        torch.save(self.vae.state_dict(), self.params['last_vae_checkpoint_path'])

        print(f'Elapsed time for training: {elapsed_time} \n')
        with open(self.params['log_path'], 'a') as f:
            f.write(f'Elapsed time for training: {elapsed_time} \n')

        plot_training_loss(train_losses, self.params)

    def test(self, test_loader, model):
        test_losses = []
        test_loss = 0
        test_mse_loss = 0
        test_kld_loss = 0

        model.eval()

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)

                x_rec, mu, log_var = model(x)
                loss, mse_loss, kld_loss = self.criterion(x_rec, x, mu, log_var)

                test_loss += loss.item()
                test_mse_loss += mse_loss.item()
                test_kld_loss += kld_loss.item()

        test_loss /= len(test_loader)
        test_mse_loss /= len(test_loader)
        test_kld_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'\n Test Set Loss: {test_loss} (MSE: {test_mse_loss}, KLD: {test_kld_loss}) \n')

    def plot_results(self, data_loader, model, model_name="vae"):

        filename = os.path.join('./plots', 'vae_mean.png')

        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                x = x.to(self.device)

                mu, log_var = model.encode(x)

                if i == 0:
                    z_mean = model.reparameterize(mu, log_var).cpu()
                    y_test = y
                else:
                    z = model.reparameterize(mu, log_var).detach().cpu()
                    z_mean = torch.cat([z_mean, z], dim=0)
                    y_test = torch.cat([y_test, y], dim=0)

        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test.numpy())
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename)
        plt.show()





