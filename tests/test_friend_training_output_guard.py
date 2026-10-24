"""Both friend CLIs accept launcher records and preserve existing experiment data."""
import json
import sys

import pytest

from cxr.phase32 import train as phase32_train
from cxr.e08_tune import train as e08_train


class StartupAccepted(Exception):
    pass


@pytest.fixture(params=[phase32_train, e08_train], ids=['phase32', 'e08_tune'])
def startup(request, monkeypatch, tmp_path):
    cli = request.param
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
    for name in ['train_stdout.log', 'train_stderr.log']:
        (run / name).write_text('')
    (run / 'train_stdout.log.pid.json').write_text(json.dumps({'pid': 123, 'host': 'test'}))
    monkeypatch.setattr(cli.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cli, 'load_config', lambda _: settings)
    monkeypatch.setattr(cli, 'output_root', lambda: output)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(config), '--run-id', 'startup_probe'])

    def accepted(*args, **kwargs):
        raise StartupAccepted

    monkeypatch.setattr(cli.shutil, 'copy2', accepted)
    return cli, run


def test_cli_accepts_live_launcher_record(startup):
    cli, _ = startup
    with pytest.raises(StartupAccepted):
        cli.main()


@pytest.mark.parametrize('name', ['last_training.pt', 'best_model.pt', 'config.yaml', 'unrelated.txt'])
def test_cli_preserves_existing_experiment_data(startup, name):
    cli, run = startup
    protected = run / name
    protected.write_bytes(b'preserve existing data')
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert protected.read_bytes() == b'preserve existing data'
