import numpy as np
from scripts.run_feature_ablation import FEATURE_VARIANTS, SelectedFeatureModel, sample_indices


class FakeRegressor:
    def predict(self, X):
        return X.sum(axis=1)


def test_feature_variants():
    assert len(FEATURE_VARIANTS['full_29']) == 29
    assert FEATURE_VARIANTS['goal_offsets_only'] == [2, 3]
    assert len(FEATURE_VARIANTS['without_wall_window']) == 4


def test_select_feature_model():
    X = np.arange(58).reshape(2, 29)
    model = SelectedFeatureModel(FakeRegressor(), [2, 3])
    assert model.predict(X).tolist() == [5, 63]


def test_sample_indices_bounds():
    rng = np.random.default_rng(42)
    result = sample_indices(20, 5, rng)
    assert len(result) == len(set(result)) == 5
    assert all(0 <= i < 20 for i in result)
