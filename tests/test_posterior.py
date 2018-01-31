import numpy as np
from numpy.testing import assert_almost_equal

from credible_interval import Posterior


def test_posterior():
    np.random.seed(20180131)
    x = np.random.normal(10, 5, 10000)
    posterior = Posterior(x)
    assert_almost_equal(9.9243057716556589, posterior.mean)
    assert_almost_equal(9.761182264312712, posterior.mode)
    assert_almost_equal(5.0345710789483089, posterior.std)
