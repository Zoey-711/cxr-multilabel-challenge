import pytest


def test_state_dict_checkpoint_loads_on_gpu_without_pretrained_weights(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU-only checkpoint validation requires CUDA")
    from cxr.models.factory import create_model
    from cxr.utils.checkpoint import check_model_file_size, load_model_state, save_model_state

    source = create_model("densenet121", pretrained=False)
    path = save_model_state(source, tmp_path / "model.pt")
    assert check_model_file_size(path) <= 500_000_000
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"format_version", "model_name", "num_classes", "model_state_dict"}
    loaded = load_model_state(path, device="cuda")
    assert next(loaded.parameters()).device.type == "cuda"
    assert torch.equal(loaded.classifier.weight.cpu(), source.classifier.weight)
