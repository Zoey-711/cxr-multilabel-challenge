"""CPU tests for auxiliary routing, export and loss normalization."""
import copy
import torch
import pytest

from cxr.models.factory import create_model
from cxr.phase34_aux import AuxiliaryDenseNet, combined_loss, gradient_probe, tensor_hash


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_main_initialization_and_logits_unchanged():
    torch.manual_seed(42)
    main = create_model("densenet121", pretrained=False).eval()
    original = tensor_hash(main.state_dict())
    rng = torch.get_rng_state().clone()
    auxiliary = AuxiliaryDenseNet(copy.deepcopy(main), True).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert tensor_hash(auxiliary.main.state_dict()) == original
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(main(x), auxiliary(x)[0])
    exported = create_model("densenet121", pretrained=False).eval()
    exported.load_state_dict(auxiliary.main.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(exported(x), auxiliary(x)[0])
    assert not any("auxiliary" in k for k in exported.state_dict())


def test_auxiliary_only_gradient_reaches_backbone():
    model = AuxiliaryDenseNet(create_model("densenet121", pretrained=False), True)
    batch = {"image": torch.randn(2, 3, 64, 64), "target": torch.randint(0, 2, (2, 10)).float()}
    result = gradient_probe(model, batch, torch.device("cpu"))
    assert result["conv0_grad_l2"] > 0
    assert result["downstream_stage_aux_only_gradient"] is False


def test_loss_and_uneven_microbatch_accumulation():
    main = torch.randn(5, 10, requires_grad=True)
    aux = torch.randn(5, 10, requires_grad=True)
    target = torch.randint(0, 2, (5, 10)).float()
    mask = torch.ones_like(target, dtype=torch.bool)
    loss, _, _ = combined_loss(main, aux, target, mask, .2, 50)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(main, target) + .2 * torch.nn.functional.binary_cross_entropy_with_logits(aux, target)
    assert torch.allclose(loss, expected)
    parts = sum(combined_loss(main[s], aux[s], target[s], mask[s], .2, 50)[0] for s in (slice(0, 3), slice(3, 5)))
    assert torch.allclose(loss, parts)
    grads = torch.autograd.grad(loss, (main, aux), retain_graph=True)
    micro = torch.autograd.grad(parts, (main, aux))
    assert all(torch.allclose(a, b) for a, b in zip(grads, micro))
    with pytest.raises(ValueError):
        combined_loss(main, None, target, mask, .2, 50)


def test_zero_alpha_matches_original_bce():
    x = torch.randn(2, 10, requires_grad=True)
    y = torch.randint(0, 2, (2, 10)).float()
    mask = torch.ones_like(y, dtype=torch.bool)
    loss, _, aux = combined_loss(x, None, y, mask, 0., 20)
    assert torch.allclose(loss, torch.nn.functional.binary_cross_entropy_with_logits(x, y))
    assert aux.item() == 0


def test_cpu_pair_analysis_aligns_ids_and_writes_report(tmp_path):
    import json
    from cxr.phase34_aux import analyze_pair, write_json, write_csv, digest
    truth = []
    predictions = []
    for i in range(8):
        positive = [j for j in range(1, 11) if (j + i) % 2 == 0]
        truth.append({"Subject_id":f"subject_{i//2}","Study_id":f"study_{i}",
                      "Predict_class":";".join(str(j) for j in positive)})
        predictions.append({"Subject_id":f"subject_{i//2}","Study_id":f"study_{i}",
                            **{f"Prob_{j}":.8 if j in positive else .2 for j in range(1,11)}})
    for task in ("AUX-BASE", "AUX-01"):
        run = tmp_path / task
        run.mkdir()
        # Reversed CSV order must still align by patient+Study IDs.
        write_csv(run/"oof_predictions.csv",predictions if task=="AUX-BASE" else predictions[::-1])
        write_csv(run/"train_history.csv",[{"epoch":i,"main_bce":1/i,"data_order_sha256":f"same-{i}"} for i in range(1,6)])
        write_json(run/"model_size.json",{"bytes":100})
        write_json(run/"DONE.json",{"state":"DONE","sha256":{p.name:digest(p) for p in run.iterdir()}})
    cfg={"epochs":5,"bootstrap_seed":42,"bootstrap_draws":20,"continue_delta":.002,"max_class_drop":.02}
    analyze_pair(cfg,tmp_path,truth)
    report=json.loads((tmp_path/"paired_bootstrap.json").read_text())
    assert report["delta"]==0 and report["base_auc"]==1
    assert report["studies"]==8 and report["patients"]==4
    assert report["ci95"]==[0,0]
    assert report["continuation_evidence_rule_met"] is False
    assert (tmp_path/"training_curve.svg").is_file()
    assert "Fold1" in (tmp_path/"final_decision.md").read_text()


def test_changed_completed_artifact_rejected(tmp_path):
    from cxr.phase34_aux import write_json, digest, verify_done
    path=tmp_path/"oof_predictions.csv"
    path.write_text("original",encoding="utf-8")
    write_json(tmp_path/"DONE.json",{"sha256":{path.name:digest(path)}})
    path.write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="Completed file changed"):
        verify_done(tmp_path)
