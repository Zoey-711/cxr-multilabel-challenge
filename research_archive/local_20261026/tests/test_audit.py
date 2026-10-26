from cxr.data.audit import ManifestSchema, audit_rows


def test_audit_counts_studies_subjects_images_and_raw_labels():
    rows = [
        {"patient": "p1", "exam": "s1", "file": "a.jpg", "finding": "1"},
        {"patient": "p1", "exam": "s1", "file": "b.jpg", "finding": ""},
        {"patient": "p1", "exam": "s2", "file": "c.jpg", "finding": "-1"},
        {"patient": "p2", "exam": "s3", "file": "d.jpg", "finding": "0"},
    ]
    report = audit_rows(rows, ManifestSchema("patient", "exam", "file", ("finding",)))
    assert (report.row_count, report.subject_count, report.study_count) == (4, 2, 3)
    assert report.studies_with_multiple_images == 1
    assert report.subjects_with_multiple_studies == 1
    assert report.label_values["finding"] == {"1": 1, "<blank>": 1, "-1": 1, "0": 1}


def test_audit_rejects_wrong_column_mapping():
    try:
        audit_rows([{"patient": "p1"}], ManifestSchema("patient", "exam", "file"))
    except ValueError as exc:
        assert "exam" in str(exc)
    else:
        raise AssertionError("Missing configured column must fail")
