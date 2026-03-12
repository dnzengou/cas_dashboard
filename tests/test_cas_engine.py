import numpy as np

from app_main import (
    UtilityParams,
    DynamicParams,
    utility_transform,
    expected_utilities_matrix,
    update_mix,
    normalized
)

def test_utility_transform_positive_negative():
    u = UtilityParams(gamma=1.0, lambda_loss=2.0)
    x = np.array([2.0, -3.0])
    out = utility_transform(x, u)

    assert out[0] == 2.0      # positive branch
    assert out[1] == -6.0     # negative branch with λ_loss applied

def test_expected_utilities_matrix_shape():
    payoff = np.array([[1,2],[3,4]])
    opp = np.array([0.4,0.6])
    u = UtilityParams()
    EU = expected_utilities_matrix(payoff, opp, u)

    assert EU.shape == (2,)
    assert isinstance(EU, np.ndarray)

def test_update_mix_replicator_validity():
    x = np.array([0.5, 0.5])
    EU = np.array([1.0, 2.0])
    dyn = DynamicParams(method="replicator", eta=0.1)

    new_x = update_mix(x, EU, dyn)

    # probabilities sum to 1
    assert np.isclose(np.sum(new_x), 1.0)

def test_update_mix_logit_softmax_valid():
    x = np.array([0.5, 0.5])
    EU = np.array([1.0, 2.0])
    dyn = DynamicParams(method="smoothed_best_response", eta=0.3, beta=2.0)

    new_x = update_mix(x, EU, dyn)
    assert np.isclose(np.sum(new_x), 1.0)

def test_normalized_vector():
    v = np.array([1.0, 1.0])
    out = normalized(v)

    assert np.allclose(out, np.array([0.5, 0.5]))