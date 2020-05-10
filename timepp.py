from numpy.random import poisson, uniform
import numpy as np
import scipy.integrate as integrate
from math import exp, log
from collections import defaultdict

__all__ = [
    "PoissonProcess",
    "HomogeneousPoissonProcess",
    "ExpPoissonProcess",
    "GraphHawkesProcess",
]


class PoissonProcess:
    """This class represents an inhomogeneous Poisson Process over the positive
    reals.

    :param function rate: The intensity or rate function of the process,
                          mapping a positive time to the value of the rate
                          function at that time. It is allowed to be ``None``
                          if the ``integral`` parameter is specified.
    :param function integral: The integral of the rate function. This parameter
                              overrides the default implementation provided 
                              by the :meth:`integral` method.
    :param function inverse: The inverse of the integral function. This
                             overrides the default implementation provided by
                             the :meth:`inverse` method.
    """

    def __init__(self, rate=None, integral=None, inverse=None):
        if rate is not None:
            self.rate = rate
        if integral is not None:
            self.integral = integral
        if inverse is not None:
            self.inverse = inverse

    def integral(self, time):
        """Returns the value of the integral of the rate function from ``0`` to
        the given ``time``. The default implementation uses
        ``scipy.integrate.quad``. If a closed-formed expression for the
        integral is known, it will usually improve performance to override this
        method by using the ``integral`` parameter of :class:`PoissonProcess`.
        """
        return integrate.quad(self.rate, 0, time)[0]

    def inverse(self, s, tmax, err=1e-10):
        """Returns the inverse of the integral function. The default
        implementation performs a bisection search. If a closed-formed for the
        inverse function is known, it is recommended to override this method by
        specifying the ``inverse`` parameter of :class:`PoissonProcess`.

        :param float s: the value to invert
        :param float t: an upper-bound on the inverted value. That is, the
                        return value is searched in the interval [0,t].
        :param float err: the precision of the bisection search. The true value
                          is guaranteed to be in an interval of length
                          `err` centered at the return value.
        """
        a, b = 0, tmax
        while b - a >= err:
            mid = (a + b) / 2.0
            if self.integral(mid) <= s:
                a = mid
            else:
                b = mid
        return (a + b) / 2.0

    def events(self, tmax):
        """Samples a random realization of the Poisson process over the interval
        ``[0,tmax]``. The return value is a ``numpy.array`` of the random event
        times, sorted by increasing time. The running time is linear in the
        number of sampled events.
        """
        it = self.integral(tmax)
        n = poisson(lam=it, size=1)[0]
        u = uniform(high=it, size=n)
        for i in range(n):
            u[i] = self.inverse(u[i], tmax)
        return np.sort(u)


class HomogeneousPoissonProcess(PoissonProcess):
    """This class represents a homogeneous Poisson process, that is, one with
    a constant rate function whose value is given by the ``rate`` parameter.
    """

    def __init__(self, rate):
        self.rate = rate

    def integral(self, t):
        ""
        return self.rate * t

    def events(self, tmax):
        ""
        n = poisson(self.rate * tmax)
        return np.sort(uniform(high=tmax, size=n))


class ExpPoissonProcess(PoissonProcess):
    """This class represents a Poisson process with an exponentially decaying
    rate function of the form :math:`\lambda(t) = \\alpha e^{-\\beta t}`.  In
    particular, the expected number of events over :math:`[0,\infty)` is
    :math:`\\alpha/\\beta`.
    """

    def __init__(self, alpha, beta):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def integral(self, t):
        ""
        return self.alpha / self.beta * (1.0 - exp(-self.beta * t))

    def inverse(self, s, t):
        ""
        return -1.0 / self.beta * log(1.0 - s * self.beta / self.alpha)

    def events(self, tmax):
        ""
        it = self.integral(tmax)
        n = poisson(lam=it, size=1)[0]
        u = uniform(high=it, size=n)
        return np.sort(
            -1.0 / self.beta * np.log(1.0 - self.beta / self.alpha * u)
        )


class GraphHawkesProcess:
    """This class represents a multivariate Hawkes Process spreading over
    a graph.

    Denote by :math:`\lambda_0` the rate function of the background process and
    by :math:`\\nu` the rate function of the *exciting* process, then the
    temporal point process occurring at each vertex :math:`v` of the graph has
    a conditional intensity function given by

    .. math::

        \lambda_v(t) = \lambda_0(t) + \sum_{t_k < t} E(v_k,
        v)\cdot\\nu(t-t_k)\,,

    where :math:`t_k` denotes the time of the :math:`k`-th event, 
    :math:`v_k` denotes the vertex at which it occurred and :math:`E(v_k, v)`
    is the binary indicator of whether an edge between :math:`v_k` and
    :math:`v` exists.

    The ``background`` and ``excitation`` parameters are used to specify the
    background and excitation processes respectively. These should be classes
    derived from :class:`PoissonProcess`, or at the very least, classes
    equipped with an `events()` method to sample random event times.

    The ``graph`` parameter should be an iterable and indexable object, such
    that ``graph[k]`` returns an iterable sequence of vertex k's neighbors. For
    example, a univariate Hawkes Process can be obtained by specifying for
    ``graph`` a single vertex with a self-loop: ``{0: [0]}``.
    """

    def __init__(self, graph, background, excitation):
        self.graph = graph
        self.background = background
        self.kernel = excitation

    def events(self, tmax):
        """Samples a random realization of the Hawkes process over the interval
        ``[0,tmax]``. The return value is a dictionary mapping each vertex to
        a sorted ``numpy.array`` of the random times of the events which
        occurred at this vertex. The running time is linear in the number of
        sampled events.
        """
        events = defaultdict(list)
        gen = dict()

        # generate background events (first generation) for each node
        for node in self.graph:
            a = self.background.events(tmax)
            if a.size > 0:
                gen[node] = a
                events[node].append(a)

        # generate triggered events generation by generation
        while gen:
            next_gen = dict()
            for node, times in gen.items():
                for time in times:
                    for neighbor in self.graph[node]:
                        # each (node, time) event from the previous generation
                        # induces a new Poisson process starting from time on
                        # each of its neighbor
                        a = time + self.kernel.events(tmax - time)
                        if a.size > 0:
                            next_gen[node] = a
                            events[node].append(a)
            gen = next_gen

        return {
            node: np.sort(np.concatenate(times))
            for (node, times) in events.items()
        }
