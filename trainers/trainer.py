import torch
from tqdm import tqdm
import time

from losses.loss import DetectionLoss
from models.detector import Detector
from utility.plotter import plot_training_loss

class TrainerDetector:
    def __init__(self, params, detector):

        self.params = params
        self.device = params['device']

        self.detector = detector
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=params['lr'])
        self.criterion = DetectionLoss()

    def train_step(self, x, y):
        self.optimizer.zero_grad()

        y_pred = self.detector(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, train_loader):
        train_losses, val_losses = [], []

        start_training = time.time()

        try:
            for epoch in tqdm(range(self.params['epochs']), total=self.params['epochs']):
                self.detector.train()
                loss, count = 0, 0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    batch_loss = self.train_step(x, y)

                    loss += batch_loss

                loss /= len(train_loader)
                train_losses.append(loss)

                print(f'Epoch: {epoch}, Loss: {loss} \n')

                with open(self.params['log_path'], 'a') as f:
                    f.write(f'Epoch: {epoch}, Loss: {loss} \n')

        except KeyboardInterrupt:
            print('Training interrupted.')

        end_training = time.time()
        elapsed_time = end_training - start_training

        torch.save(self.detector.state_dict(), self.params['last_model_checkpoint_path'])

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





