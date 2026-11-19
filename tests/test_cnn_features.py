import logging
import pytest
import pandas as pd
import numpy as np

from sqlalchemy import and_

from API_operations.helpers.Service import Service
from BO.Prediction import DeepFeatures
from DB.CNNFeatureVector import ObjectCNNFeatureVector
from DB.Image import Image
from DB.Object import ObjectHeader
from DB.Project import Project
from DB.Acquisition import Acquisition
from DB.Sample import Sample
from data.db_load import OBJECTS_IN_PROJECT


def test_cnn_features_storage(database):
    prj_id = 1
    prj = Project(projid=prj_id)
    with Service() as sce:
        ret = DeepFeatures.find_missing(sce.session, prj)
        assert len(ret) == OBJECTS_IN_PROJECT
        obj_ids = list(ret.keys())

    # Prepare fake CNN features to insert
    features = []
    for i, oi in enumerate(obj_ids):
        features.append([(i + 1) * 0.1] * 50)
    features_df = pd.DataFrame(features, index=obj_ids)

    # Test features insertion
    with Service() as sce:
        n_inserts = DeepFeatures.save(sce.session, features_df)
        assert n_inserts == OBJECTS_IN_PROJECT
        sce.session.commit()

    # Test features retrieval
    with Service() as sce:
        ret_feats = DeepFeatures.np_read_for_objects(sce.session, obj_ids)
        assert (ret_feats == np.array(features, dtype="float32")).all()

    # Test find_missing without any missing features
    with Service() as sce:
        ret_missing = DeepFeatures.find_missing(sce.session, prj)
        assert ret_missing == {}

    # Test deletion
    with Service() as sce:
        # Delete all features manually, the deletion code is only in Web app
        sub_qry = sce.session.query(ObjectHeader.objid).join(
            Acquisition, Acquisition.acquisid == ObjectHeader.acquisid
        ).join(
            Sample, and_(Sample.sampleid == Acquisition.acq_sample_id, Sample.projid == prj_id)
        )
        n_deletes = sce.session.query(ObjectCNNFeatureVector).filter(
            ObjectCNNFeatureVector.objcnnid.in_(sub_qry)
        ).delete(synchronize_session=False)
        assert n_deletes == OBJECTS_IN_PROJECT
        sce.session.commit()

    # Test find_missing after deletion
    with Service() as sce:
        ret_missing_post = DeepFeatures.find_missing(sce.session, prj)
        assert len(ret_missing_post) == OBJECTS_IN_PROJECT

    # Test features retrieval for missing, should raise an error
    with Service() as sce:
        with pytest.raises(AssertionError):
            DeepFeatures.np_read_for_objects(sce.session, obj_ids)


def test_cnn_features_save_idempotent(database):
    prj_id = 1
    prj = Project(projid=prj_id)
    with Service() as sce:
        ret = DeepFeatures.find_missing(sce.session, prj)
        assert len(ret) == OBJECTS_IN_PROJECT
        obj_ids = list(ret.keys())

    features = [[(i + 1) * 0.1] * 50 for i in range(len(obj_ids))]
    features_df = pd.DataFrame(features, index=obj_ids)
    partial_df = features_df.iloc[:2]

    # Insert a subset first (simulating another prediction job on a shared source project)
    with Service() as sce:
        n_first = DeepFeatures.save(sce.session, partial_df)
        assert n_first == len(partial_df)
        sce.session.commit()

    # Saving the full set again must be idempotent and not raise UniqueViolation
    with Service() as sce:
        n_second = DeepFeatures.save(sce.session, features_df)
        assert n_second == OBJECTS_IN_PROJECT - len(partial_df)
        sce.session.commit()

    # Saving the full set a third time inserts 0 new rows
    with Service() as sce:
        n_third = DeepFeatures.save(sce.session, features_df)
        assert n_third == 0
        sce.session.commit()

    # All features are readable and intact
    with Service() as sce:
        ret_feats = DeepFeatures.np_read_for_objects(sce.session, obj_ids)
        assert (ret_feats == np.array(features, dtype="float32")).all()


def test_find_missing_uses_cnn_image_position(database):
    prj_id = 2
    with Service() as sce:
        # Pick one object in project 2 and add images at non-contiguous imgrank values
        # (existing image has imgrank=1; add imgrank=0 before it and imgrank=5 after it)
        first_obj_id = sorted(
            DeepFeatures.find_missing(sce.session, Project(projid=prj_id)).keys()
        )[0]
        existing_img = (
            sce.session.query(Image)
            .filter(Image.objid == first_obj_id, Image.imgrank == 1)
            .one()
        )
        img_rank0 = Image(
            objid=first_obj_id,
            imgrank=0,
            width=100,
            height=100,
            orig_file_name="rank0.png",
        )
        img_rank5 = Image(
            objid=first_obj_id,
            imgrank=5,
            width=100,
            height=100,
            orig_file_name="rank5.jpeg",
        )
        sce.session.add_all([img_rank0, img_rank5])
        sce.session.flush()
        try:
            expected_rank0 = img_rank0.img_to_file()
            expected_rank1 = existing_img.img_to_file()
            expected_rank5 = img_rank5.img_to_file()

            # Default (None) -> 1st image (imgrank=0)
            ret_default = DeepFeatures.find_missing(
                sce.session, Project(projid=prj_id, cnn_image_position=None)
            )
            assert ret_default[first_obj_id] == expected_rank0
            assert len(ret_default) == OBJECTS_IN_PROJECT

            # Position 1 -> 1st image (imgrank=0)
            ret_pos1 = DeepFeatures.find_missing(
                sce.session, Project(projid=prj_id, cnn_image_position=1)
            )
            assert ret_pos1[first_obj_id] == expected_rank0
            assert len(ret_pos1) == OBJECTS_IN_PROJECT

            # Position 2 -> 2nd image (imgrank=1); other objects only have 1 image
            ret_pos2 = DeepFeatures.find_missing(
                sce.session, Project(projid=prj_id, cnn_image_position=2)
            )
            assert ret_pos2 == {first_obj_id: expected_rank1}

            # Position 3 -> 3rd image (imgrank=5)
            ret_pos3 = DeepFeatures.find_missing(
                sce.session, Project(projid=prj_id, cnn_image_position=3)
            )
            assert ret_pos3 == {first_obj_id: expected_rank5}
        finally:
            sce.session.rollback()
