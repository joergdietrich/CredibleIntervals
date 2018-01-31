#!/usr/bin/env python

from __future__ import print_function, division

import sys

import numpy as np

from scipy import integrate, optimize, stats


class Posterior:
    def __init__(self, samples):
        self.kde = stats.gaussian_kde(samples)
        self.mean = samples.mean()
        self.std = samples.std()
        negative_posterior = lambda x: -1 * self.kde.evaluate(x)
        self.mode = optimize.fmin(negative_posterior, self.mean,
                                  disp=False)[0]
        self.ymax = self.kde.evaluate(self.mode)
        xextra = 0.5 * (samples.max() - samples.min())
        self.xmin = samples.min() - xextra
        self.xmax = samples.max() + xextra
        self.low_lim = self.xmin
        self.high_lim = self.xmax
        return

    def evaluate(self, x):
        return self.kde.evaluate(x)


def integral_posterior_limited_support_eq_value(fvalue,
                                                posterior,
                                                value):
    return integral_posterior_limited_support(fvalue, posterior) \
        - value


def integral_posterior_limited_support(fvalue, posterior):
    # search left and right of x0 with brentq to find point where
    # posterior has value. Then use these points a limits.
    if fvalue == 0:
        return 1
    if fvalue == posterior.ymax:
        return 0
    posterior_eq_val = lambda x, fvalue: posterior.evaluate(x) - fvalue
    low_lim = optimize.brentq(posterior_eq_val,
                              posterior.xmin,
                              posterior.mode,
                              args=(fvalue,))
    high_lim = optimize.brentq(posterior_eq_val,
                               posterior.mode,
                               posterior.xmax,
                               args=(fvalue,))
    posterior.low_lim = low_lim
    posterior.high_lim = high_lim
    val = integrate.quad(posterior.evaluate, low_lim, high_lim)
    return val[0]


def cut_from_top(posterior, level):
    val = optimize.brentq(integral_posterior_limited_support_eq_value,
                          0, posterior.ymax,
                          args=(posterior, level))
    return val


def credible_intervals(posterior, levels=(0.95, 0.68)):
    tot = integrate.quad(posterior.evaluate, posterior.xmin,
                         posterior.xmax)[0]
    if not np.isclose(tot, 1):
        raise ValueError("Posterior not normalized to 1", tot)
    limits = []
    for level in levels:
        if level > tot:
            raise ValueError("desired value exceeds area under posterior",
                             level)
        _ = cut_from_top(posterior, level)
        limits.append(posterior.low_lim)
        limits.append(posterior.high_lim)
    return sorted(limits)


def credible_intervals_chain(chain, levels=(0.95, 0.68), silent=False):
    levels = list(levels)
    limit_list = []
    level_keys = []
    levels.sort(reverse=True)
    if silent is False:
        print("Dim\tMean\tMode\tStdDev", end='')
    for sign in [-1, 1]:
        for level in levels:
            level_key = '{0:.2f}%'.format(level * 100 * sign)
            level_keys.append(level_key)
            if silent is False:
                print("\t{:s}".format(level_key), end='')
        levels.sort()
    if silent is False:
        print("\n===============================================================")
    for i, samples in enumerate(chain.T):
        limit_list.append(dict())
        try:
            posterior = Posterior(samples)
            limits = credible_intervals(posterior, levels)
            if silent is False:
                print("{0:2d}\t{1: 6.3f}\t{2: 6.3f}\t{3: 6.3f}".format(i,
                      posterior.mean,
                      posterior.mode,
                      posterior.std),
                      end='')
            limit_list[i]['mean'] = posterior.mean
            limit_list[i]['mode'] = posterior.mode
            limit_list[i]['std'] = posterior.std
            for j, lim in enumerate(limits):
                limit_list[i][level_keys[j]] = lim
                if silent is False:
                    print("\t{0: 6.3f}".format(lim), end='')
        except:
            if silent is False:
                print("{0:2d}\t\tskipped (multi-modal, flat, or delta?)".format(i),
                      end='')
            limit_list[i]['mean'] = np.nan
            limit_list[i]['mode'] = np.nan
            limit_list[i]['std'] = np.nan
            for level_key in level_keys:
                limit_list[i][level_key] = np.nan
        if silent is False:
            print("")
    return limit_list


if __name__ == "__main__":
    chain = np.loadtxt(sys.argv[1])
    limits = credible_intervals_chain(chain)
