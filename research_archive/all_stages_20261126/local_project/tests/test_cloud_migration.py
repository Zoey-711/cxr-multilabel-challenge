"""Fast synthetic checks for split ZIPs, path overrides and portable bundle."""

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from cxr.data.archive import archive_path, read_split_rows, split_archive_path
from cxr.data.dataset import CXRArchiveDataset
from cxr.utils.cloud_paths import output_root, pretrained_home
from scripts.cloud_preflight import verify_pack
from scripts.cloud_run_fold import config_file
from scripts.pack_cloud_data import pack_split, sha256
from scripts.prepare_cloud_bundle import bundle_entries


def test_split_zip_loading_and_manifest(tmp_path, monkeypatch):
    original = tmp_path / "combined.zip"
    image = io.BytesIO()
    Image.new("L", (20, 10), 120).save(image, format="JPEG")
    with zipfile.ZipFile(original, "w") as archive:
        for split in ("train", "valid"):
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=["Dicom_id", "Subject_id", "Study_id", "ViewPosition",
                                                         "PatientOrientation", "Rows", "Columns", "Predict_class"])
            writer.writeheader()
            writer.writerow({"Dicom_id": "pic_1", "Subject_id": "subject_1", "Study_id": "study_1",
                             "ViewPosition": "AP", "PatientOrientation": "Erect", "Rows": "10",
                             "Columns": "20", "Predict_class": "1;10"})
            archive.writestr(f"cxr_first/{split}.csv", out.getvalue())
            archive.writestr(f"cxr_first/{split}/subject_1/study_1/pic_1.jpg", image.getvalue())
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest = {"format": "cxr_split_zip_v1", "files": {}}
    with zipfile.ZipFile(original) as archive:
        for name, split in (("train.zip", "train"), ("valid.zip", "valid")):
            items = [x for x in archive.infolist() if x.filename.startswith(f"cxr_first/{split}")]
            path = pack / name
            pack_split(archive, items, path)
            manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest_path = pack / "pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("CXR_DATA_ROOT", str(pack))
    assert archive_path({"data": {"archive_path": str(original)}}) == pack
    assert split_archive_path(pack, "val") == pack / "valid.zip"
    assert len(read_split_rows(pack, "train")) == 1
    assert CXRArchiveDataset(pack, "val")[0]["target"].nonzero().flatten().tolist() == [0, 9]
    assert len(verify_pack(pack, manifest_path)) == 2
    (pack / "valid.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        verify_pack(pack, manifest_path)


def test_cloud_roots_are_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("CXR_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("CXR_PRETRAINED_ROOT", raising=False)
    assert output_root() == Path("outputs")
    assert pretrained_home(tmp_path) == tmp_path / ".wheels/torch_cache"
    monkeypatch.setenv("CXR_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("CXR_PRETRAINED_ROOT", str(tmp_path / "weights"))
    assert output_root() == tmp_path / "outputs"
    assert pretrained_home(tmp_path) == tmp_path / "weights"
    monkeypatch.setenv("CXR_OUTPUT_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        output_root()


def test_bundle_sanitizes_paths_without_changing_protocol():
    root = Path(__file__).resolve().parents[1]
    entries = bundle_entries(root)
    configs = {name: content for name, content in entries.items() if name.startswith("configs/")}
    assert len(configs) == 15
    assert all(b"archive_path: null" in content for content in configs.values())
    assert all(b"F:/" not in content and b"F:\\" not in content for content in configs.values())
    fold_report = json.loads(entries["outputs/analysis/phase17_patient_folds/fold_report.json"])
    assert fold_report["source_archive"] is None
    assert config_file(root, "D17", 1).name == "f17_densenet121_fold1_384.yaml"
    with pytest.raises(ValueError):
        config_file(root, "D17", 6)
