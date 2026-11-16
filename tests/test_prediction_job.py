import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

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
