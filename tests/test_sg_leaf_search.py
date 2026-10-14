import numpy as np
import pandas as pd

from analytics.sg_leaf_search import _active_batch, excess_paths, information_ratio
from strategy.index_enhance import build_books, fit_active, result_frame, review_sessions


def test_a_steadier_excess_scores_a_higher_information_ratio():
    calm = np.full(40, 0.001) + np.linspace(-0.0002, 0.0002, 40)
    wild = np.resize([0.02, -0.01], 40)
    assert information_ratio(calm) > information_ratio(wild)


def test_batch_active_matches_the_single_name_fit():
    index_w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    scores = pd.Series([1.0, -1.0, 0.2], index=index_w.index)
    covariance = pd.DataFrame(np.eye(3) * 0.0004, index=index_w.index, columns=index_w.index)
    single = fit_active(index_w, scores, covariance)
    batch = _active_batch(index_w.to_numpy(), scores.to_numpy()[None, :], covariance.to_numpy())
    np.testing.assert_allclose(batch[0], single.to_numpy(), atol=1e-8)


def test_the_review_book_earns_the_next_session():
    days = pd.bdate_range("2024-03-01", "2024-03-22")
    prices = pd.DataFrame(
        {"A": np.linspace(10, 12, len(days)), "B": np.linspace(10, 9, len(days))},
        index=days,
    )
    shares = pd.Series({"A": 1.0, "B": 1.0})
    scores = pd.DataFrame({"A": 1.0, "B": -1.0}, index=days)
    index_book, port_book, _reviews = build_books(prices, shares, scores)
    results = result_frame(index_book, port_book, prices)
    cube = np.zeros((1, len(days), 2))
    cube[0, :, 0] = 1.0
    cube[0, :, 1] = -1.0
    excess = excess_paths(prices, shares, cube, review_sessions(days))
    aligned = pd.Series(excess[0], index=days).reindex(results.index)
    np.testing.assert_allclose(aligned.to_numpy(), results["excess_return"].to_numpy(), atol=1e-8)
