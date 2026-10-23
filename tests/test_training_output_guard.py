"""The real CLI accepts launcher bookkeeping but protects existing training data."""
import json
import sys

import pytest

from scripts import batch_train


class StartupAccepted(Exception):
    pass


def prepare(monkeypatch, tmp_path):
    config = tmp_path / 'input.yaml'
    config.write_text('fixture config\n')
    settings = {
        'model': {'name': 'densenet121', 'pretrained': 'imagenet'},
        'training': {},
        'validation': {'primary_metric': 'macro_auc'},
    }
    output = tmp_path / 'output'
    run = output / 'experiments' / 'startup_probe'
    run.mkdir(parents=True)
    (run / 'train_stdout.log').write_text('')
    (run / 'train_stderr.log').write_text('')
    (run / 'train_stdout.log.pid.json').write_text(json.dumps({'pid': 123, 'host': 'test'}))
    monkeypatch.setattr(batch_train.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(batch_train, 'load_config', lambda _: settings)
    monkeypatch.setattr(batch_train, 'output_root', lambda: output)
    monkeypatch.setattr(sys, 'argv', ['batch_train.py', '--config', str(config),
                                    '--run-id', 'startup_probe'])

    def accepted(*args, **kwargs):
        raise StartupAccepted

    # Stop immediately after the directory guard, before model/data/GPU work.
    monkeypatch.setattr(batch_train.shutil, 'copy2', accepted)
    return run


def test_cli_starts_with_the_live_launcher_pid_record(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    with pytest.raises(StartupAccepted):
        batch_train.main()


@pytest.mark.parametrize('name', ['last_training.pt', 'best_model.pt', 'config.yaml', 'unrelated.txt'])
def test_cli_still_refuses_preexisting_experiment_data(monkeypatch, tmp_path, name):
    run = prepare(monkeypatch, tmp_path)
    protected = run / name
    protected.write_bytes(b'preserve existing data')
    with pytest.raises(SystemExit) as error:
        batch_train.main()
    assert error.value.code == 2
    assert protected.read_bytes() == b'preserve existing data'
