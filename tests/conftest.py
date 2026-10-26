import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import pytest
import torch

@pytest.fixture(autouse=True)
def private_paths(tmp_path, monkeypatch):
    import core, train, api
    data = tmp_path / 'data'
    data.mkdir()
    values = dict(DATA=data, AUDIO=data/'recordings', META=data/'metadata.jsonl',
                  OUTPUT=data/'generated', CKPT=tmp_path/'checkpoints'/'latest.pt')
    for module in (core, train, api):
        for key, value in values.items():
            if hasattr(module, key):
                monkeypatch.setattr(module, key, value)
    for module in (train, api):
        for key, value in dict(STATUS=data/'training_status.json', STATUS_LOCK=data/'training_status.lock', LOCK=data/'training.lock', DATA_LOCK=data/'dataset.lock').items():
            if hasattr(module, key):
                monkeypatch.setattr(module, key, value)
    torch.set_num_threads(2)
