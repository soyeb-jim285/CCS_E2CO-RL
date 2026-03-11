"""V17 Fine-Tune + Structured ODE — re-export V14 loss (6 terms: 5 data + eigenvalue reg)."""

from versions.v14_latent_physics.loss import StructuredODELoss as FinetuneODELoss
