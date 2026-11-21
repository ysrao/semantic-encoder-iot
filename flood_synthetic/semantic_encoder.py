"""Semantic-communications on-device encoder — REFERENCE IMPLEMENTATION.

Status: reference build, NOT a trained model from the semantic-mIoT-NTN study.
  The underlying work (github.com/ysrao/semantic-miot-ntn, duplicated as
  ysrao/ntn-eval) is a trace-driven simulation in numpy/pandas whose
  device-side semantic function is a DETERMINISTIC RULE — threshold +
  persistence + hysteresis + causal smoothing (schemes.py run_semantic_agri /
  run_semantic_env, stage3.rule_below / rule_above) emitting a compact task
  report (20-byte application payload vs 200 raw). There is no neural encoder
  and no weights in that work; see ../INVENTORY.md.
  The concept note (17 Nov 2026), section 2, specifies the deployed form of
  that function as "a small neural network on the sensor, the meter, or the
  handset". This module implements that deployed form's architecture.
  Weights are randomly initialised (seeded) and UNTRAINED; training the
  encoder to reproduce/extend the rule's behaviour on the study's traces is
  future work.

What it does (per the concept note):
  The encoder runs on-device and transmits only the semantic content the task
  requires; the NTN link carries a fraction of the raw traffic; the receiving
  side (task decoder and collector) is unchanged. Sized to an IoT SoC budget
  (Snapdragon / Dragonwing class), int8-quantised via Qualcomm AI Hub.

Input: float32 tensor (batch, 64) — a sensing window of 16 timesteps x 4
  channels, row-major (timestep-major), each channel normalised:
    ch0 primary_obs    soil moisture (agri) or river level (env), normalised
                       against its task scale (field capacity / danger level)
    ch1 threshold      the task threshold in the same normalised scale
                       (crop-stage SM threshold / 4.5 m danger level)
    ch2 rainfall       rainfall intensity, normalised
    ch3 temperature    temperature, normalised
  Window length 16 covers the simulation's causal smoothing windows
  (8 samples agri at 15-min step; 6 samples env at 5-min step) with headroom.

Output: float32 tensor (batch, 8) — the semantic latent, tanh-bounded.
  8 float32 values = 32 bytes raw, 8 bytes int8-quantised — inside the
  simulation's 20-byte semantic application-payload budget once quantised
  and packed with the task/confidence header the receiver already expects.
"""

import torch
import torch.nn as nn

WINDOW = 16
CHANNELS = 4
INPUT_DIM = WINDOW * CHANNELS  # 64
LATENT_DIM = 8

CHANNEL_NAMES = ["primary_obs", "threshold", "rainfall", "temperature"]


class SemanticEncoder(nn.Module):
    """Compact MLP encoder: 16x4 sensing window -> 8-dim semantic latent."""

    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_example_input(batch: int = 1) -> torch.Tensor:
    """Representative input: env task, level rising through the danger threshold.

    16 timesteps x [primary_obs (level/4.5m, clipped), threshold (=1.0),
    rainfall, temperature], flattened row-major.
    """
    levels = torch.linspace(0.78, 1.08, WINDOW)          # crossing 1.0 = danger
    thr = torch.ones(WINDOW)
    rain = torch.linspace(0.2, 0.9, WINDOW)
    temp = torch.full((WINDOW,), 0.55)
    window = torch.stack([levels, thr, rain, temp], dim=-1)  # (16, 4)
    return window.reshape(1, INPUT_DIM).repeat(batch, 1)


if __name__ == "__main__":
    torch.manual_seed(20261117)
    model = SemanticEncoder().eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"SemanticEncoder parameters: {n}")
    with torch.no_grad():
        out = model(build_example_input(2))
    print(f"input  (2, {INPUT_DIM}) -> output {tuple(out.shape)}")
    print(f"latent[0] = {out[0].tolist()}")
    print("(untrained — values are meaningless until trained on the study traces)")
