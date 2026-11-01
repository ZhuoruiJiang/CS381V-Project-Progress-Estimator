"""Brightness-based dummy: exercises the pipeline, not a learned task progress model."""

import torch

from coffee_progress.evaluation import CallableEstimator, Observation


class DummyRewardModel(torch.nn.Module):
    """Return an arbitrary score from image pixels, without timestamps or labels."""

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(frames.mean(dim=(-3, -2, -1)))


def create_estimator(device: str = "cpu") -> CallableEstimator:
    """Replace model loading and predict below to adapt your own checkpoint."""
    model = DummyRewardModel().to(device).eval()

    def predict(observation: Observation) -> float:
        latest_frames = torch.stack([frames[-1] for frames in observation.images.values()]).to(
            device
        )
        return model(latest_frames).mean().item()

    return CallableEstimator(predict)
