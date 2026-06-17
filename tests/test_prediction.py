import logging
import pytest
import pandas as pd
import numpy as np

from API_operations.helpers.Service import Service
from BO.Prediction import DeepFeatures

def test_prediction_functions():

    obj_ids = [5,6,7,8,9,10,11,12]
    assert len(obj_ids) == 8

    # Prepare fake CNN features to insert
    features = list()
    for i, oi in enumerate(obj_ids):
        features.append([(i + 1) * 0.1] * 50)
    features_df = pd.DataFrame(features, index=obj_ids)

    # Test features insertion
    with Service() as sce:
        n_inserts = DeepFeatures.save(sce.session, features_df)
        assert n_inserts == 8
        sce.session.commit()

    # Test features retrieval
    with Service() as sce:
        ret = DeepFeatures.np_read_for_objects(sce.session, obj_ids)
        assert (ret == np.array(features, dtype="float32")).all()

    # Test find_missing without any missing features
    with Service() as sce:
        ret = DeepFeatures.find_missing(sce.session, prj_id)
        assert ret == {}

    # Test deletion
    with Service() as sce:
        n_deletes = DeepFeatures.delete_all(sce.session, prj_id)
        assert n_deletes == 8
        sce.session.commit()

    # Test find_missing after deletion
    with Service() as sce:
        ret = DeepFeatures.find_missing(sce.session, prj_id)
        assert len(ret) == 8

    # Test features retrieval in empty table, should raise an error
    with Service() as sce:
        with pytest.raises(AssertionError):
            ret = DeepFeatures.np_read_for_objects(sce.session, obj_ids)
