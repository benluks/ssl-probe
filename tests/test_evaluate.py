import json

import pytest
import torch
from torch import nn

from ssl_probe.commands.evaluate import read_config, regression_metrics, restore_models
from ssl_probe.probe import Probe


def test_metrics_and_undefined_values():
    scores = regression_metrics([1, 2, 3], [1, 2, 3])
    assert scores["r2"] == pytest.approx(1)
    assert scores["ccc"] == pytest.approx(1)
    assert regression_metrics([1, 1], [2, 2])["r2"] is None
    assert regression_metrics([1], [2])["pearson"] is None
    assert regression_metrics([1, 2, 3], [2, 2, 2])["r2"] == pytest.approx(0)


def test_config_requires_architecture_and_supports_wandb(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "model.ckpt"
    with pytest.raises(ValueError, match="configuration"):
        read_config(checkpoint)
    config = {"encoder": "wavlm", "target": "jitter", "context_size": 3}
    (tmp_path / "config.yaml").write_text(json.dumps({k: {"value": v} for k, v in config.items()}))
    assert read_config(checkpoint) == config


class Encoder(nn.Module):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def test_restore_finetuned_weights_and_normalization(monkeypatch):
    monkeypatch.setattr(
        "ssl_probe.commands.evaluate.build_content_encoder", lambda *args: Encoder()
    )
    probe = Probe(6, 1, [4], "gelu")
    config = {
        "encoder": "fake",
        "target": "jitter",
        "context_size": 3,
        "hidden_dim": [4],
        "train_encoder": True,
        "target_normalization": "standardize",
        "target_standardization": {"mean": 2.0, "std": 3.0, "count": 10},
    }
    state = {"probe." + k: v for k, v in probe.state_dict().items()}
    with pytest.raises(ValueError, match="missing encoder"):
        restore_models({"state_dict": state}, config)
    state["content_encoder.weight"] = torch.ones(1)
    encoder, restored, target = restore_models({"state_dict": state}, config)
    assert encoder.weight.item() == 1
    assert not encoder.training and not restored.training
    x = torch.randn(5, 3, 2)
    assert torch.equal(probe(x), restored(x))
    assert torch.equal(restored(x), restored(x))
    assert target.task.compute(torch.zeros(2, 1), torch.ones(2)).prediction.tolist() == [2.0, 2.0]
    del state["probe.network.0.weight"]
    with pytest.raises(RuntimeError):
        restore_models({"state_dict": state}, config)


def test_evaluation_writes_aligned_outputs(monkeypatch, tmp_path):
    import csv
    from types import SimpleNamespace

    from quick_convert.data import AudioSample
    from quick_convert.data.resources import ResourceCollection, ResourceRef

    from ssl_probe.commands import evaluate
    from ssl_probe.targets import TARGETS

    source = AudioSample(utt_id="max_phonation/1234", path=tmp_path / "1234.wav")
    frames = [
        AudioSample(
            utt_id=f"{source.utt_id}:{index:06d}",
            path=source.path,
            resources=ResourceCollection.from_refs(
                [
                    ResourceRef(name="content", kind="torch_tensor", value=torch.ones(3, 2)),
                    ResourceRef(
                        name="jitter", kind="torch_tensor", value=torch.tensor([float(index)])
                    ),
                ]
            ),
        )
        for index in (1, 3)
    ]

    class Dataset:
        content_frame_hz = 50
        base_dataset = [source]

        def __init__(self, *args, **kwargs):
            assert args[2] == tmp_path / "features"
            assert kwargs["context_size"] == 3

        def extract_frames(self, samples):
            assert len(samples) == 1
            return frames

    config = {"encoder": "fake", "target": "jitter", "context_size": 3}
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: SimpleNamespace(
            checkpoint=tmp_path / "model.ckpt",
            config=None,
            device="cpu",
            smile_root=tmp_path / "features",
            dataset="clac",
            split="max_phonation",
            manifest=None,
            root=str(tmp_path),
            spk_id_template=None,
            speaker_stats=None,
            frame_batch_size=1,
            out_dir=tmp_path / "out",
        ),
    )
    monkeypatch.setattr(evaluate, "read_config", lambda *args: config)
    torch.save({"state_dict": {}}, tmp_path / "model.ckpt")
    probe = Probe(6, 1)
    monkeypatch.setattr(
        evaluate, "restore_models", lambda *args: (Encoder(), probe.eval(), TARGETS["jitter"])
    )
    monkeypatch.setattr(evaluate, "FrameDataset", Dataset)
    evaluate.main()
    metrics = json.loads((tmp_path / "out/metrics.json").read_text())
    assert metrics["pooled"]["count"] == 2
    assert metrics["per_speaker"]["1234"]["count"] == 2
    with (tmp_path / "out/predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["time_seconds"]) for row in rows] == [0.02, 0.06]
    assert (tmp_path / "out/scatter.png").is_file()
