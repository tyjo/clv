import numpy as np
import scipy.stats
import sys


def idx_to_rank_sum(idx, n):
    """Map from an integer idx to a value in the support of
    the Wilcoxon test statistic.

    Specifically, if S is the support of the test statistic, 
    then this is a map from
        {0, ..., |S| - 1} ->  S

    Parameters
    ----------
        idx : integer
            An integer from {0, ..., |S| - 1}.
        n : integer
            The number of paired samples.

    Returns
    -------
        rank_sum : integer
            The rank_sum corresponding to idx.

    Examples
    --------
        If we have n = 3 paired samples, then the
        map is given by
          idxes = {  0  1  2  3  4  5  6 }
                     |  |  |  |  |  |  |
            S   = { -6 -4 -2  0  2  4  6 }
    """
    support_size = int(n*(n+1)/2 + 1)

    if idx >= support_size:
        raise ValueError("Outside of support")

    shift = 1 if support_size % 2 == 0 else 0
    middle = (support_size-1)/2
    rank_sum = 2*(idx-np.floor(middle)) - shift
    rank_sum = int(rank_sum)
    return rank_sum


def rank_sum_to_idx(rank_sum, n):
    """Map from a rank sum in the support of the Wilcoxon
    test statistic to an integer idx.

    This is the inverse mapping of idx_to_rank_sum.

    Specifically, if S is the support of the test statistic, 
    then this is a map from
        S -> {0, ..., |S| - 1} 

    Parameters
    ----------
        rank_sum : integer
            An integer in the support of the test statistic
        n : integer
            The number of paired samples.

    Returns
    -------
        idx : integer
            The idx in the support corresponding to rank_sum.

    Examples
    --------
        If we have n = 3 paired samples, then the
        map is given by
            S   = { -6 -4 -2  0  2  4  6 }
                     |  |  |  |  |  |  |
          idxes = {  0  1  2  3  4  5  6 }

    """
    if rank_sum < -n*(n+1)/2 or rank_sum > n*(n+1)/2:
        raise ValueError("Outside of support")

    support_size = int(n*(n+1)/2 + 1)
    shift = 1 if support_size % 2 == 0 else 0
    middle = (support_size-1)/2
    for i in range(int(support_size)):
        if rank_sum == 2*(i-np.floor(middle)) - shift:
            return i


def compute_counts(n):
    """Recursively counts the coefficients of each term in the mgf.

    The mgf is given by
        mgf_X(t) = 1/2^n \prod_{j=1}^n (e^{-tj} + e^{tj})
                 = 1/2^n \sum_{j=1}^n c_j e^{tj}
                 = \sum_{j=1}^n Pr(X = j) e^{tj}

    This function computes c_j for a number of paired samples n.

    By expanding and collecting e^{tj} for each j, we can compute
    the pmf of the Wilcoxon test statistic. We can do this recursively
    by noticing that
        1/2^n \prod_{j=1}^n (e^{-tj} + e^{tj})
        = 1/2^n (e^{-nt} + e^{nt}) \prod_{j=1}^{n-1} (e^{-tj} + e^{tj})
    Thus, if we have the coefficients c_j for n-1, we can compute the
    coefficients c_j for n by expanding the bottom equation and counting
    e^{tj} for each j.
.

    References
    ----------
        [1] Hogg, Robert V., Joseph McKean, and Allen T. Craig. Introduction 
        to Mathematical Statistics. 7th Edition. pp. 541-544.
    """
    if n == 1:
        counts = np.array([1, 1])
        return counts
    else:
        counts = compute_counts(n-1)
        support_size = int(n*(n+1)/2 + 1)
        new_counts = np.zeros(support_size)
        for i in range(counts.size):
            rank_sum = idx_to_rank_sum(i, n-1)
            new_rank_sum1 = rank_sum + n
            new_rank_sum2 = rank_sum - n

            min_rank = -n*(n+1)/2
            max_rank = n*(n+1)/2

            if counts[i] > 0:
                if new_rank_sum1 <= max_rank:
                    new_counts[rank_sum_to_idx(new_rank_sum1, n)] += counts[i]

                if new_rank_sum2 >= min_rank:
                    new_counts[rank_sum_to_idx(new_rank_sum2, n)] += counts[i]
        return new_counts



def compute_pmf(n):
    """Compute the support and pmf given n paired samples.
    
    Parameters
    ----------
        n : integer
            The number of paired samples
    Returns
    -------
        support : numpy array
            An array giving the support of the pmf
        pmf : numpy array
            An array giving the probability of each integer in the support
    """
    support_size = int(n*(n+1)/2 + 1)
    support = np.array([idx_to_rank_sum(i, n) for i in range(support_size)])
    pmf = compute_counts(n)/np.power(2,n)
    assert np.abs(pmf.sum() - 1) < 1E-8, pmf.sum()
    return support, pmf



def wilcoxon_exact(x, y=None, alternative="two-sided"):
    """
    Calculate the Wilcoxon signed-rank test statistic and exact p-values.
    
    Given matched samples, x_i and y_i, the Wilcoxon signed-rank test tests the
    null that x_i - y_i is symmetric around zero. In practice, it is used to test
    whether x_i and y_i are from the same population with different location 
    parameters.

    There are several different versions of the test statistic. The one used here
    is
        T = sign(z_1) R|z_1| + ... + sign(z_n) R|z_n|
    where
        z_i = x_i - y_i       if y_i is specified
        z_i = x_i             otherwise.

    The pmf has no closed form, but for small sample sizes it is possible to compute
    the pmf by solving for the coefficients of the moment generating function.

    Parameters
    ----------
        x : array_like
            The first set of data values (if y is specified) or the difference
            between two sets of data values
        y : array_like optional
            If specified, the difference x - y is used for the test statistic
        alternative : {"two-sided", "greater", "less"}, optional
            The alternative hypothesis tested.
            If "two-sided", the test is
                x_i - y_i > 0 or x_i - y_i < 0
            If "greater", the test it
                x_i - y_i > 0
            If "less", the test is
                x_i - y_i < 0

    Returns
    -------
        T : float
            The test-statistic.
        p : float
            The p-value.


    Examples
    --------
    >>> import numpy as np
    >>> from wilcoxon_exact import wilcoxon_exact
    >>> x = np.array([1.83,  0.50,  1.62,  2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    >>> y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
    >>> wilcoxon_exact(x, y, alternative="greater")
    (35.0, 0.01953125)

    >>> x = np.array([-6.1, 4.3, 7.2, 8.0, -2.1])
    >>> wilcoxon_exact(x, alternative="two-sided")
    (7.0, 0.4375)
    """
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("Alternative must be either 'two-sided'", "'greater' or 'less'")

    x = np.array(x)
    if x.ndim > 1:
        raise ValueError("Sample x must be one-dimensional")

    if y is not None:
        y = np.array(y)
        if y.ndim > 1:
            raise ValueError("Sample y must be one-dimensional")
        if x.shape != y.shape:
            raise ValueError("Sample x and y must have the same length.")
        diff = x - y
    else:
        diff = x

    if np.unique(np.abs(diff)).size != diff.size:
        if y is None:
            raise ValueError("abs(x) values must be unique")
        else:
            raise ValueError("abs(x - y) values must be unique")

    ranks = scipy.stats.rankdata(np.abs(diff))
    signs = np.sign(diff)
    T = (signs*ranks).sum()

    n = diff.size
    if n > 30:
        print("warning: sample size is large for exact calculation\n" + 
              "         calculation may be slow", file=sys.stderr)
    rank_sum, pmf = compute_pmf(n)

    if alternative == "less":
        idx = rank_sum <= T
        p = pmf[idx].sum()
    elif alternative == "greater":
        idx = rank_sum >= T
        p = pmf[idx].sum()
    else:
        idx = np.logical_or(rank_sum <= -np.abs(T), rank_sum >= np.abs(T))
        p = pmf[idx].sum()

    return T, p


if __name__ == "__main__":
    print("Running tests...")
    # Test the mapping from array indices to the support
    n = 1
    support_size = int(n*(n+1)/2 + 1)
    support = np.array([-1, 1])
    for i in range(support_size):
        rank_sum = idx_to_rank_sum(i, n)
        assert rank_sum == support[i], str(rank_sum) + " is not == " + str(support[i])

    n = 2
    support_size = int(n*(n+1)/2 + 1)
    support = np.array([-3, -1, 1, 3])
    for i in range(support_size):
        rank_sum = idx_to_rank_sum(i, n)
        assert rank_sum == support[i], str(rank_sum) + " is not == " + str(support[i])

    n = 3
    support_size = int(n*(n+1)/2 + 1)
    support = np.array([-6, -4, -2, 0, 2, 4, 6])
    for i in range(support_size):
        rank_sum = idx_to_rank_sum(i, n)
        assert rank_sum == support[i], str(rank_sum) + " is not == " + str(support[i])

    n = 4
    support_size = int(n*(n+1)/2 + 1)
    support = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    for i in range(support_size):
        rank_sum = idx_to_rank_sum(i, n)
        assert rank_sum == support[i], str(rank_sum) + " is not == " + str(support[i])

    # Test the mapping from the support to indices
    n = 1
    support = np.array([-1, 1])
    for idx, rs in enumerate(support):
        i = rank_sum_to_idx(rs, n)
        assert i == idx

    n = 2
    support = np.array([-3, -1, 1, 3])
    for idx, rs in enumerate(support):
        i = rank_sum_to_idx(rs, n)
        assert i == idx

    n = 3
    support = np.array([-6, -4, -2, 0, 2, 4, 6])
    for idx, rs in enumerate(support):
        i = rank_sum_to_idx(rs, n)
        assert i == idx

    n = 4
    support = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    for idx, rs in enumerate(support):
        i = rank_sum_to_idx(rs, n)
        assert i == idx


    # Test computing the pmf
    counts = compute_counts(1)
    expected_counts = np.array([1., 1.])
    assert np.all(counts == expected_counts)

    counts = compute_counts(2)
    expected_counts = np.array([1, 1, 1, 1])
    assert np.all(counts == expected_counts)

    counts = compute_counts(3)
    expected_counts = np.array([1, 1, 1, 2, 1, 1, 1])
    assert np.all(counts == expected_counts)

    counts = compute_counts(4)
    expected_counts = np.array([1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1])
    assert np.all(counts == expected_counts)

    rank_sum, pmf = compute_pmf(3)
    expected_rank_sum = np.array([-6, -4, -2, 0, 2, 4, 6])
    expected_pmf = compute_counts(3)/8
    assert np.all(rank_sum == expected_rank_sum)
    assert np.all(pmf == expected_pmf)

    rank_sum, pmf = compute_pmf(4)
    expected_rank_sum = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    expected_pmf = compute_counts(4)/16
    assert np.all(rank_sum == expected_rank_sum)
    assert np.all(pmf == expected_pmf)


    # Compare p-values to those generated by R
    x = np.array([1.83,  0.50,  1.62,  2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
    p_greater = 0.01953125
    p_less = 0.986328125
    p_two_tailed = 0.0390625
    assert np.abs(p_greater - wilcoxon_exact(x, y, "greater")[1]) < 1E-8
    assert np.abs(p_less - wilcoxon_exact(x, y, "less")[1]) < 1E-8
    assert np.abs(p_two_tailed - wilcoxon_exact(x, y, "two-sided")[1]) < 1E-8

    x = np.array([-6.1, 4.3, 7.2, 8.0, -2.1])
    p_greater = 0.21875
    p_less = 0.84375
    p_two_tailed = 0.4375
    assert np.abs(p_greater - wilcoxon_exact(x, alternative="greater")[1]) < 1E-8
    assert np.abs(p_less - wilcoxon_exact(x, alternative="less")[1]) < 1E-8
    assert np.abs(p_two_tailed - wilcoxon_exact(x, alternative="two-sided")[1]) < 1E-8

    x = np.array([ 0.02378057, -0.68174183,  1.11878671,  0.34532977, -0.6687004 ,
                  -0.00275748,  0.61589287, -0.28073615,  2.32557442,  0.057978  ,
                  -0.16706318,  0.94763388,  1.49308607,  0.61162639, -0.09053253,
                  -0.13645805, -0.96742671, -0.3784958 ,  1.21554404,  0.08509771,
                   0.92971309,  0.2383083 ,  0.78747881,  0.82000769,  0.53554306])

    y = np.array([-0.27826135, -0.5497125 ,  1.12381066, -0.33309764,  0.30702763,
                  -1.10632243, -0.08192293,  0.32331644,  1.04667997,  0.56002087,
                  -0.88494827,  1.84830939, -2.10739496,  0.02836235,  0.30636346,
                  -0.77036039, -0.21900135,  0.20526964,  0.55322561, -0.0153534 ,
                  -0.1933098 , -0.73011513,  0.48943407,  0.30204235, -1.29466832])
    p_greater = 0.03131321073
    p_less = 0.9706242383
    p_two_tailed = 0.06262642145
    assert np.abs(p_greater - wilcoxon_exact(x, y, "greater")[1]) < 1E-8
    assert np.abs(p_less - wilcoxon_exact(x, y, "less")[1]) < 1E-8
    assert np.abs(p_two_tailed - wilcoxon_exact(x, y, "two-sided")[1]) < 1E-8

    x = np.array([ 0.2553419 , -0.3687772 ,  1.38766691,  0.66001067,  1.06808852,
                  -0.13471264, -0.64539139, -0.53209119, -0.57985291, -0.55703202,
                   0.52719496, -0.71739911,  1.46841159,  0.83334944,  1.51441298,
                   1.78476198, -0.30565303,  0.27826273,  1.04988288,  0.41767253,
                   0.08049793, -0.39440538,  0.53596691,  0.43917537,  1.67664441,
                  -1.5416117 , -0.59672013, -0.10630783, -0.17345269, -1.3209056 ])
    p_greater = 0.1642349288
    p_less = 0.8408019599
    p_two_tailed = 0.3284698576
    assert np.abs(p_greater - wilcoxon_exact(x, alternative="greater")[1]) < 1E-8
    assert np.abs(p_less - wilcoxon_exact(x, alternative="less")[1]) < 1E-8
    assert np.abs(p_two_tailed - wilcoxon_exact(x, alternative="two-sided")[1]) < 1E-8
    print("Passed all tests!")

