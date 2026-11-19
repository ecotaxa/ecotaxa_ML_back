import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from psycopg2.errors import DeadlockDetected, UniqueViolation
from sqlalchemy.exc import IntegrityError, OperationalError

from API_models.filters import ProjectFiltersDict
from API_models.prediction import PredictionReq
from API_operations.helpers.Service import Service
from BO.Job import JobBO
from DB.Job import Job, DBJobStateEnum
from helpers import DateTime

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from API_operations.GPU_Prediction import GPUPredictForProject


def create_DB_job(session, job_type: str, user_id: int, args: dict) -> int:
    """
    Create a job directly in the DB.
    """
    job = Job()
    job.state = DBJobStateEnum.Pending
    job.progress_msg = JobBO.PENDING_MESSAGE
    job.creation_date = job.updated_on = DateTime.now_time()
    job.type = job_type
    job.owner_id = user_id
    job.params = json.dumps(args)
    job.inside = job.reply = json.dumps({})
    job.messages = json.dumps([])
    session.add(job)
    session.flush([job])
    job_id = job.id
    session.commit()
    return job_id


def test_prediction_job(database):
    features = ["obj.depth_min", "obj.depth_max", "fre.area"]
    req: PredictionReq = PredictionReq(
        project_id=1, source_project_ids=[2, 3], features=features
    )
    filters: ProjectFiltersDict = {}
    job = GPUPredictForProject(req, filters)
    with Service() as sce:
        job.job_id = create_DB_job(sce.session, job.JOB_TYPE, 1, job.init_args({}))
    job.do_background()


class _FakeResult:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.rowcount = len(rows)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.rows)


def test_classify_exact_chunk_size_does_not_predict_empty_chunk(monkeypatch):
    job = GPUPredictForProject.__new__(GPUPredictForProject)
    job.CHUNK_SIZE = 3  # 10K in original bug
    job.req = SimpleNamespace(use_scn=False, project_id=1)
    job.ro_session = MagicMock()
    job.session = MagicMock()
    job.update_progress = MagicMock()

    tgt_res = _FakeResult([(1, None, 1.0), (2, None, 2.0), (3, None, 3.0)])
    classifier = MagicMock()
    classifier.predict_all.return_value = ([1, 1, 1], [1.0, 1.0, 1.0])

    target_obj_set = MagicMock()
    target_obj_set.classify_auto_mult.return_value = (3, {})
    monkeypatch.setattr(
        "API_operations.GPU_Prediction.EnumeratedObjectSet",
        MagicMock(return_value=target_obj_set),
    )
    monkeypatch.setattr(
        "API_operations.GPU_Prediction.ProjectBO.update_taxo_stats", MagicMock()
    )

    nb_changes = job.classify(MagicMock(), tgt_res, classifier, ["fre.area"], {})

    assert nb_changes == 3
    classifier.predict_all.assert_called_once()
    target_obj_set.classify_auto_mult.assert_called_once()


def test_prediction_req_normalizes_pre_mapping_keys_and_values():
    req = PredictionReq(
        project_id=1,
        source_project_ids=[2],
        features=["fre.area"],
        pre_mapping={"30815": "1001", 85069: 1002, "bad": "x"},
    )

    assert req.pre_mapping[30815] == 1001
    assert req.pre_mapping[85069] == 1002
    assert "bad" not in req.pre_mapping


@pytest.mark.parametrize("orig_exc", [UniqueViolation(), DeadlockDetected()])
def test_classify_conflict_rolls_back_and_sets_job_error(monkeypatch, orig_exc):
    job = GPUPredictForProject.__new__(GPUPredictForProject)
    job.CHUNK_SIZE = 2
    job.req = SimpleNamespace(use_scn=False, project_id=1)
    job.ro_session = MagicMock()
    job.session = MagicMock()
    job.update_progress = MagicMock()
    job.set_job_result = MagicMock()

    tgt_res = _FakeResult([(1, None, 1.0), (2, None, 2.0), (3, None, 3.0)])
    classifier = MagicMock()
    classifier.predict_all.return_value = ([1, 1], [1.0, 1.0])

    err_cls = (
        IntegrityError if isinstance(orig_exc, UniqueViolation) else OperationalError
    )
    db_err = err_cls("stmt", {}, orig_exc)

    target_obj_set = MagicMock()
    target_obj_set.classify_auto_mult.side_effect = [
        (2, {}),
        db_err,
    ]
    monkeypatch.setattr(
        "API_operations.GPU_Prediction.EnumeratedObjectSet",
        MagicMock(return_value=target_obj_set),
    )
    update_stats_mock = MagicMock()
    monkeypatch.setattr(
        "API_operations.GPU_Prediction.ProjectBO.update_taxo_stats",
        update_stats_mock,
    )

    ret = job.classify(MagicMock(), tgt_res, classifier, ["fre.area"], {})

    assert ret is None
    job.session.commit.assert_called_once()
    job.session.rollback.assert_called_once()
    update_stats_mock.assert_not_called()
    job.set_job_result.assert_called_once()
    kwargs = job.set_job_result.call_args.kwargs
    assert kwargs["errors"] == ["Another prediction runs on same data"]
    assert kwargs["infos"] == {"status": "error", "rowcount": 2}


def test_classify_reraises_non_conflict_db_error(monkeypatch):
    job = GPUPredictForProject.__new__(GPUPredictForProject)
    job.CHUNK_SIZE = 2
    job.req = SimpleNamespace(use_scn=False, project_id=1)
    job.ro_session = MagicMock()
    job.session = MagicMock()
    job.update_progress = MagicMock()
    job.set_job_result = MagicMock()

    tgt_res = _FakeResult([(1, None, 1.0)])
    classifier = MagicMock()
    classifier.predict_all.return_value = ([1], [1.0])

    target_obj_set = MagicMock()
    target_obj_set.classify_auto_mult.side_effect = OperationalError(
        "stmt", {}, RuntimeError("connection lost")
    )
    monkeypatch.setattr(
        "API_operations.GPU_Prediction.EnumeratedObjectSet",
        MagicMock(return_value=target_obj_set),
    )

    with pytest.raises(OperationalError):
        job.classify(MagicMock(), tgt_res, classifier, ["fre.area"], {})

    job.set_job_result.assert_not_called()


def test_ensure_deep_features_passes_project_orm_objects(monkeypatch):
    from DB.Project import Project

    job = GPUPredictForProject.__new__(GPUPredictForProject)
    job.req = SimpleNamespace(source_project_ids=[2, 3])
    src_prj_2 = Project(projid=2, title="Src 2", instrument_id="Zooscan")
    src_prj_3 = Project(projid=3, title="Src 3", instrument_id="Zooscan")
    tgt_prj = Project(
        projid=1,
        title="Target",
        instrument_id="Zooscan",
        cnn_network_id="SCN_zooscan_group1",
    )

    qry = MagicMock()
    qry.filter.return_value.all.return_value = [src_prj_2, src_prj_3]
    job.ro_session = MagicMock()
    job.ro_session.query.return_value = qry

    called_with = []

    def fake_ensure_for(project, model_name):
        called_with.append((project, model_name))
        return f"OK for {project.projid}"

    monkeypatch.setattr(job, "_ensure_deep_features_for", fake_ensure_for)

    job.ensure_deep_features(tgt_prj)

    assert called_with == [
        (tgt_prj, "SCN_zooscan_group1"),
        (src_prj_2, "SCN_zooscan_group1"),
        (src_prj_3, "SCN_zooscan_group1"),
    ]
    job.ro_session.query.assert_called_once_with(Project)
    job.ro_session.rollback.assert_called_once()


def test_ensure_deep_features_for_accepts_project_orm_object(monkeypatch):
    from DB.Project import Project

    job = GPUPredictForProject.__new__(GPUPredictForProject)
    job.ro_session = MagicMock()
    job.session = MagicMock()
    job.vault = MagicMock()
    job.models_dir = MagicMock()

    prj = Project(projid=42, title="Test", instrument_id="Zooscan")
    find_missing_mock = MagicMock(return_value={})
    monkeypatch.setattr(
        "API_operations.GPU_Prediction.DeepFeatures.find_missing",
        find_missing_mock,
    )

    diag = job._ensure_deep_features_for(prj, "SCN_zooscan_group1")

    assert diag == "All CNN present for 42"
    find_missing_mock.assert_called_once_with(job.ro_session, prj)
    job.ro_session.rollback.assert_called_once()
