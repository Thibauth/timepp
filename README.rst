Python package to perform random sampling of Poisson and Hawkes processes.

Installation
============

Directly from GitHub using ``pip``:

.. code-block:: bash

    pip install git+https://github.com/Thibauth/timepp

Example
=======

Consider a standard multivariate Hawkes Process with constant background rate
and exponentially decaying kernel. In other words the conditional intensity
function of vertex :math:`v` is given by:

.. math::

        \lambda_v(t) = \mu + \alpha\sum_{t_k < t} E(v_k, v)e^{-\beta(t-t_k)}\cdot\,,

where :math:`t_k` denotes the time of the :math:`k`-th event, 
:math:`v_k` denotes the vertex at which it occurred and :math:`E(v_k, v)`
is the binary indicator of whether an edge between :math:`v_k` and
:math:`v` exists. This random process can be simulated over the time interval
:math:`[0,10]` with the following code.

.. code-block:: python

    from timepp import (HomogeneousPoissonProcess, GraphHawkesProcess,
                        ExpPoissonProcess)

    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    mu, alpha, beta = 0.5, 1, 5
    background = HomogeneousPoissonProcess(mu)
    excitation = ExpPoissonProcess(alpha, beta)
    hp = GraphHawkesProcess(graph, background, excitation)
    print(hp.events(10))
