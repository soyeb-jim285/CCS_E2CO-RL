"""V13 Physics Fine-Tuning loss — same as V1's 8-term FD physics loss."""

from versions.v1_physics_constrained.loss import PhysicsLoss  # noqa: F401

PhysicsFinetuneLoss = PhysicsLoss
