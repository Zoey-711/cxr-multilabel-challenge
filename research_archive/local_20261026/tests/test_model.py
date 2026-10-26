import pytest


def test_densenet121_structure_without_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not torch.cuda.is_available():
        pytest.skip("GPU-only model validation requires CUDA")
    from cxr.models.factory import create_model

    model = create_model("densenet121", num_classes=10, pretrained=False).to("cuda")
    assert model.classifier.in_features == 1024
    assert model.classifier.out_features == 10
    with torch.no_grad():
        output = model.eval()(torch.zeros(1, 3, 64, 64, device="cuda"))
        torch.cuda.synchronize()
    assert tuple(output.shape) == (1, 10)


def test_convnext_tiny_structure_without_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not torch.cuda.is_available():
        pytest.skip("GPU-only model validation requires CUDA")
    from cxr.models.factory import create_model

    model = create_model("convnext_tiny", num_classes=10, pretrained=False).to("cuda").eval()
    assert model.classifier[-1].out_features == 10
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64, device="cuda"))
    assert tuple(output.shape) == (1, 10)


def test_swin_tiny_structure_without_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not torch.cuda.is_available():
        pytest.skip("GPU-only model validation requires CUDA")
    from cxr.models.factory import create_model, imagenet_weight_info

    model = create_model("swin_t", num_classes=10, pretrained=False).to("cuda").eval()
    assert model.head.in_features == 768
    assert model.head.out_features == 10
    assert imagenet_weight_info("swin_t")[2] == "swin_t-704ceda3.pth"
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64, device="cuda"))
    assert tuple(output.shape) == (1, 10)
    assert torch.isfinite(output).all()
