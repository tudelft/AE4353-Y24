import torch
from torch import nn


class BaseAE(nn.Module):
    def __init__(self, latent_dim, encoder, decoder):
        super(BaseAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, _ = self.encoder(x)
        recon_x = self.decoder(mu)

        return {"x": x, "mu": mu, "recon_x": recon_x}

    def loss_fn(self, x, mu, recon_x):
        # TODO: Implement the loss function (reconstruction loss)

        return torch.tensor(0.)

    def reconstruct(self, x):
        return self(x)["recon_x"]

    def embed(self, x):
        mu, _ = self.encoder(x)
        return mu

    def predict(self, x):
        return self.decoder(self.embed(x))

    def interpolate(self, x_start, x_end, granularity=10):
        assert (
            x_start.shape == x_end.shape
        ), "x_start and x_end must have the same shape"

        z_start = self.embed(x_start)
        z_end = self.embed(x_end)

        alphas = torch.linspace(0, 1, granularity).view(-1, 1, 1).to(z_start.device)

        interp_z = z_start * (1 - alphas) + z_end * alphas

        decoded_z = self.decoder(interp_z.view(-1, self.latent_dim))
        return decoded_z


class BetaVAE(BaseAE):
    def __init__(self, latent_dim, beta, encoder, decoder):
        super(BetaVAE, self).__init__(latent_dim, encoder, decoder)

        self.beta = beta

    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)

        z, _ = self.sample_latent(mu, std)

        recon_x = self.decoder(z)

        return {"x": x, "mu": mu, "log_var": log_var, "recon_x": recon_x}

    @staticmethod
    def sample_latent(mu, std):
        """
        Sample from the latent space using the reparameterization trick.
        Returns:
            z: torch.Tensor, the latent space sample
            eps: torch.Tensor, the noise used for sampling
        """
        # TODO: Reparameterization trick
        return torch.tensor(0.), torch.tensor(0.)

    def loss_fn(self, x, mu, log_var, recon_x):
        # TODO: Implement the loss function (reconstruction loss + beta * KL divergence)
        return torch.tensor(0.)

