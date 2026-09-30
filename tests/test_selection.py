import pandas as pd

from cer_ae.selection import gradient_plus_random_clients


def test_gradient_plus_random_selection_is_disjoint_and_reproducible():
    metadata = pd.DataFrame({"ID": range(1000), "gradient": [i - 500 for i in range(1000)]})
    cfg = {"gradient_ranking": "absolute", "n_gradient_clients": 300,
           "n_random_clients": 200, "n_clients": 500, "seed": 42}
    first = gradient_plus_random_clients(range(1000), metadata, cfg)
    second = gradient_plus_random_clients(range(1000), metadata, cfg)
    assert len(first) == first.ID.nunique() == 500
    assert first.ID.tolist() == second.ID.tolist()
    top = first[first.selection_reason == "top_absolute_gradient"]
    assert len(top) == 300
    assert top.gradient.abs().min() >= metadata.loc[~metadata.ID.isin(top.ID), "gradient"].abs().max()
