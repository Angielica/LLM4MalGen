import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, p_dims, q_dims=None):
        super().__init__()

        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must be equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        tmp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(tmp_q_dims[:-1], tmp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.dropout = nn.Dropout(0.2)

        self.init_weights()

    def init_weights(self):
        for layer in self.q_layers:
            nn.init.xavier_normal_(layer.weight)

        for layer in self.p_layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def encode(self, x):
        mu, log_var = None, None
        h = x
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
                h = self.dropout(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                log_var = h[:, self.q_dims[-1]:]

        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
                h = self.dropout(h)
        return torch.tanh(h)

    def generate(self, n_samples=1):
        device = next(self.parameters()).device

        with torch.no_grad():
            z = torch.randn((n_samples, self.p_dims[0])).to(device)
            return self.decode(z), z

    def generate_mu_sigma(self, x):
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            return self.decode(z), z