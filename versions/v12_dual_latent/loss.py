"""V12 Dual Latent loss — same 9-term structure as V5 (architecture handles isolation)."""

from versions.v5_coordinate_pinn.loss import CoordinatePINNLoss  # noqa: F401

# The dual latent architecture handles gradient isolation between z_data and z_phys.
# The loss function is identical to V5's 9-term loss.
DualLatentLoss = CoordinatePINNLoss
