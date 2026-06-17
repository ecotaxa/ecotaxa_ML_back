# -*- coding: utf-8 -*-
# This file is part of Ecotaxa, see license.md in the application root directory for license informations.
# Copyright (C) 2015-2021  Picheral, Colin, Irisson (UPMC-CNRS)
#
# Predict classification on a project. In details, launch a job which will:
# - If requested by the user and possible, compute DeepFeatures on the source projects
# - Use selected features on source projects to train a Random Forest classifier
# - Use the trained classifier on the target project.
#
# Here is just the job registering part, the rest is in GPU_Prediction class.
#
from pathlib import Path
from typing import cast, List, Dict, Any, Optional

ProjectFiltersDict = Dict[str, Any]


class PredictionReq:
    def __init__(self, **kwargs):
        self.project_id: int = kwargs.get("project_id", 0)
        self.source_project_ids: List[int] = kwargs.get("source_project_ids", [])
        self.learning_limit: Optional[int] = kwargs.get("learning_limit")
        self.features: List[str] = kwargs.get("features", [])
        self.categories: List[int] = kwargs.get("categories", [])
        self.use_scn: bool = kwargs.get("use_scn", False)
        self.pre_mapping: Dict[int, int] = kwargs.get("pre_mapping", {})

    def dict(self):
        return self.__dict__


class PredictionRsp:
    def __init__(
        self, job_id: int = 0, errors: List[str] = None, warnings: List[str] = None
    ):
        self.job_id = job_id
        self.errors = errors or []
        self.warnings = warnings or []


# from BO.Rights import RightsBO, Action
from FS.MachineLearningModels import SavedModels
from FS.Vault import Vault
from helpers.DynamicLogs import get_logger, LogsSwitcher

# TODO: Move somewhere else
from .helpers.JobService import JobServiceBase, ArgsDict
# from .helpers.Service import Service

logger = get_logger(__name__)


class PredictForProject(JobServiceBase):
    """ """

    JOB_TYPE = "Prediction"

    def __init__(self, req: PredictionReq, filters: ProjectFiltersDict):
        super().__init__()
        self.req = req
        self.filters: ProjectFiltersDict = filters
        self.out_path: Path = Path("")
        self.vault = Vault(self.config.vault_dir())
        self.models_dir = SavedModels(self.config)

    # def run(self, current_user_id: UserIDT) -> PredictionRsp:
        """
        Initial creation, do security and consistency checks, then create the job.
        """
        # _user, _project = RightsBO.user_wants(
        #     self.session, current_user_id, Action.ANNOTATE, self.req.project_id
        # )
        # TODO: more checks, e.g. deep features models consistency
        # Security OK, create pending job
        # self.create_job(self.JOB_TYPE, current_user_id)
        # ret = PredictionRsp(job_id=self.job_id)
        # return ret

    def init_args(self, args: ArgsDict) -> ArgsDict:
        args["req"] = self.req.dict()
        args["filters"] = self.filters
        return args

    @staticmethod
    def deser_args(json_args: ArgsDict) -> None:
        json_args["req"] = PredictionReq(**json_args["req"])
        json_args["filters"] = cast(ProjectFiltersDict, json_args["filters"])

    def do_background(self) -> None:
        """
        Background part of the job.
        """
        with LogsSwitcher(self):
            self.do_prediction()

    def do_prediction(self) -> None: ...


# class PredictionDataService(Service):
#     """
#     Available models service.
#     """
#
#     def get_models(self) -> List[str]:
#         return SavedModels(self.config).list()
