from app_main import adjust_weights_for_coalitions

def test_adjust_weights_for_coalitions_basic():
    W = {("A","B"): 1.0, ("B","A"): 2.0, ("A","C"): 3.0}
    coalitions = {"MyBloc": ["A","B"]}
    beta = 2.0

    W_new = adjust_weights_for_coalitions(W, coalitions, beta)

    # internal coalition links multiplied by beta
    assert W_new[("A","B")] == 1.0 * beta
    assert W_new[("B","A")] == 2.0 * beta

    # links to external nodes unchanged
    assert W_new[("A","C")] == 3.0