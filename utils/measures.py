import numpy as np
import cvxpy as cp


def is_a_measure(w):
    return np.all((w >= 0) & (w <= 1))


def is_a_distribution(w):
    if not is_a_measure(w):
        return False
    if np.sum(w) == 1:
        return True
    else:
        return False


def density(w):
    return np.sum(w)/np.size(w)


def kl_div(w, v):
    result = 0
    for i in range(len(w)):
        if w[i] != 0:
            result += w[i] * np.log2(w[i]/v[i]) - w[i] + v[i]
        else:
            result += v[i]
    return result


def renyi_div(w, v, alpha):
    result = 0
    for i in range(len(w)):
        result += (w[i]**alpha) / (v[i]**(1-alpha))
    return (1/(alpha-1)) * np.log(result)


def bregman_projection_cvx(w, density_parameter, distribution=True):

    """
        Bregman projection for a discrete distribution or a discrete bounded measure, using CVX package
        Parameters
        ----------
        w : array-like, dtype=float, shape=n
        Discrete probability distribution or discrete bounded measure
        density_parameter: float
        Projection parameter alpha or gamma (based on being a distribution or a measure)
        distribution: bool
        Determines whether w should be treated as a distribution
    """

    n = len(w)
    V = cp.Variable(shape=n)
    W = cp.Parameter(shape=n)
    density_parameter = cp.Constant(value=density_parameter)

    W.value = np.array(w)

    objective = cp.Minimize(cp.sum(cp.kl_div(V, W)))
    if distribution:
        constraints = [V >= density_parameter / n,
                       cp.sum(V) == 1.0]
    else:
        constraints = [cp.sum(V) >= density_parameter * n,
                       V <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.status, V.value


def distributions_bregman_projection(w, alpha):

    """
        Bregman projection for a discrete distribution, based on the algorithm by Herbster and Warmuth
        Parameters
        ----------
        w : array-like, dtype=float, shape=n
        Discrete probability distribution
        alpha: float
        Projection parameter alpha
    """

    n = len(w)
    W = np.array(range(n))
    Cs = 0
    Cp = 0
    omega = 0
    s = np.sum(w)

    while len(W) > 0:
        omega = np.median([w[i] for i in W])
        L = np.array([i for i in W if w[i] < omega])
        Ls = len(L)
        Lp = np.sum([w[i] for i in L])
        M = np.array([i for i in W if w[i] == omega])
        Ms = len(M)
        Mp = np.sum([w[i] for i in M])
        H = np.array([i for i in W if w[i] > omega])
        m0 = (s - (Cs + Ls)*(alpha/n))/(s - (Cp + Lp))
        if omega*m0 < alpha/n:
            Cs = Cs + Ls + Ms
            Cp = Cp + Lp + Mp
            if len(H) == 0:
                omega = np.min([w[i] for i in range(n) if w[i] > omega])
            W = H
        else:
            W = L

    m0 = (s - Cs*(alpha/n))/(s - Cp)
    v = np.zeros(np.shape(w))

    for i in range(n):
        if w[i] < omega:
            v[i] = alpha/n
        else:
            v[i] = w[i]*m0

    return v


def approximate_bregman_projection(w, kappa, epsilon=0.0001, upper_bound=10000000):

    """
           Approximate Bregman projection for a discrete measure, based on the paper by Barak, Hardt and Kale
           Parameters
           ----------
           w : array-like, dtype=float, shape=n
           Discrete bounded measure
           kappa: float
           Density parameter kappa
           epsilon: float
           Precision parameter epsilon
           upper_bound: float
           Determines the upper-bound for the value of c

    """

    w = np.array(w)
    c = (1 + upper_bound)/2
    lower_bound = 1
    if density(w) > kappa:
        return 1
    while True:
        # print(c)
        w_s = np.clip(w * c, a_min=0, a_max=1)
        mu = density(w_s)
        # print("C is ", c, " and mu is ", mu)
        if mu > (1 + epsilon) * kappa - epsilon * kappa / 3:
            upper_bound = c
            c = (c + lower_bound) / 2

        elif mu < kappa - epsilon * kappa / 3:
            lower_bound = c
            c = (c + upper_bound) / 2
        else:
            break
    return c


def measures_bregman_projection(w, kappa):

    """
           Bregman projection for a discrete measure
           Parameters
           ----------
           w : array-like, dtype=float, shape=n
           Discrete bounded measure
           kappa: float
           Density parameter kappa

    """

    w = np.array(w)
    if density(w) >= kappa:
        return w
    w_sorted = np.sort(w)[::-1]
    n = len(w)
    c = 1
    for i in range(n):
        delta_0 = density(w_sorted[i:])
        delta = (kappa * n - i)/(n - i)
        c_i = delta/delta_0
        if c_i * w_sorted[i] <= 1:
            c = c_i
            break
    return np.clip(w * c, a_min=0, a_max=1)


def _density(sample_measure, X):
    np.sum(sample_measure) / _num_samples(X)
                
def _check_sample_measure(sample_measure, X, dtype=None):
    """Validate sample measures.

    Note that passing sample_measure=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_measure is not None:
        sample_measure = _check_sample_measure(...)

    Parameters
    ----------
    sample_measure : {ndarray, Number or None}, shape (n_samples,)
       Input sample weights.

    X : nd-array, list or sparse matrix
        Input data.

    dtype: dtype
       dtype of the validated `sample_measure`.
       If None, and the input `sample_measure` is an array, the dtype of the
       input is preserved; otherwise an array with the default numpy dtype
       is be allocated.  If `dtype` is not one of `float32`, `float64`,
       `None`, the output will be of dtype `float64`.

    Returns
    -------
    sample_measure : ndarray, shape (n_samples,)
       Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_measure is None:
        sample_measure = np.ones(n_samples, dtype=dtype)
    elif isinstance(sample_measure, numbers.Number):
        sample_measure = np.full(n_samples, sample_measure, dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_measure = check_array(
            sample_measure, accept_sparse=False, ensure_2d=False, dtype=dtype,
            order="C"
        )
        if sample_measure.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_measure.shape != (n_samples,):
            raise ValueError("sample_measure.shape == {}, expected {}!"
                             .format(sample_measure.shape, (n_samples,)))
    return sample_measure
