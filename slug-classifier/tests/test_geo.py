import pandas as pd
import pytest
import torch

from slugid.geo import combine, geo_log_probs, observations, train_geo


def test_no_location_means_image_only():
    logits = torch.randn(3, 5)
    assert torch.equal(combine(logits, None), logits)


def test_floor_limits_how_far_location_can_push_down():
    logits = torch.zeros(1, 2)
    geo = torch.tensor([[-50.0, -0.1]])
    out = combine(logits, geo, weight=1.0, floor=-8.0)
    assert out[0].tolist() == pytest.approx([-8.0, -0.1])


def test_observations_dedup_and_drop_missing_coords():
    m = pd.DataFrame(
        {
            "observation_uuid": ["a", "a", "b", "c", "d"],
            "label": ["x", "x", "y", "x", "z"],
            "split": ["train"] * 5,
            "latitude": [1.0, 1.0, None, 2.0, 3.0],
            "longitude": [1.0, 1.0, 1.0, 2.0, 3.0],
        }
    )
    obs = observations(m, "train", ["x", "y"])
    assert obs["observation_uuid"].tolist() == ["a", "c"]


def test_geo_model_learns_separate_ranges():
    rng = torch.Generator().manual_seed(0)
    n = 400
    obs = pd.DataFrame(
        {
            "label": ["north"] * n + ["south"] * n,
            "latitude": torch.cat(
                [50 + torch.randn(n, generator=rng), -30 + torch.randn(n, generator=rng)]
            ).numpy(),
            "longitude": (10 * torch.randn(2 * n, generator=rng)).numpy(),
        }
    )
    model = train_geo(
        obs, ["north", "south"], epochs=20, batch_size=128, device=torch.device("cpu")
    )
    lp = geo_log_probs(model, [50.0, -30.0], [0.0, 0.0])
    assert lp.argmax(1).tolist() == [0, 1]
