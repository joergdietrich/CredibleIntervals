import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from credible_interval import credible_intervals, credible_intervals_chain, Posterior

intervals = [[-0.12920204242099748, 4.691686659944319,
              14.85143515198398, 19.8704874290323],
             [1.7396192781790167, 2.115742137774251,
              3.1944671047284614, 3.8858295094242643],
             ]


def test_intervals():
    np.random.seed(20180131)
    p = Posterior(np.random.normal(10, 5, 10000))
    limits = credible_intervals(p)
    assert_array_almost_equal(limits, intervals[0])


def test_intervals_chain():
    np.random.seed(20180131)
    x1 = np.random.normal(10, 5, 10000)
    x2 = np.random.lognormal(1, 0.2, 10000)
    chain = np.vstack((x1, x2)).T
    limits = credible_intervals_chain(chain)
    keys = ['-95.00%', '-68.00%',
            '68.00%', '95.00%']
    for i, limit_dict in enumerate(limits):
        check_dict = dict(zip(keys, intervals[i]))
        for key in limit_dict.keys():
            if key in ["mean", "mode", "std"]:
                continue
            assert_almost_equal(check_dict[key], limit_dict[key])


def test_singular_chain():
    x = np.ones(100).reshape(-1, 1)
    limits = credible_intervals_chain(x)[0]
    for key in limits.keys():
        assert (limits[key] is np.nan)
