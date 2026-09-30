import torch

from cer_ae.models import ConditionalAutoencoder, ConvAutoencoder1D, DenseAutoencoder, VariationalAutoencoder


def test_model_output_dimensions():
    x48 = torch.randn(3, 48); x336 = torch.randn(3, 336)
    assert DenseAutoencoder(48, 4)(x48).shape == x48.shape
    assert ConvAutoencoder1D(336, 4)(x336).shape == x336.shape
    assert ConditionalAutoencoder(48, 3, 4)(x48, torch.randn(3, 3)).shape == x48.shape
    reconstruction, mu, logvar = VariationalAutoencoder(48, 4)(x48)
    assert reconstruction.shape == x48.shape and mu.shape == logvar.shape == (3, 4)
