.. xomega documentation master file, created by
   sphinx-quickstart on Fri Oct 19 11:42:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xomega's documentation!
==================================
xomega is a Python package for solving the
`generalized omega equation <https://journals.ametsoc.org/doi/abs/10.1175/1520-0493(2000)128%3C2270:MAAOAC%3E2.0.CO%3B2>`_
which takes the form

.. math::

    N^2\nabla_\text{h}w_\text{b} + f_0^2\frac{\partial^2w_\text{b}}{\partial z^2} = \beta\frac{\partial b}{\partial x} + \nabla_\text{h}\cdot{\bf Q}(u,v,\theta,\Phi).

Given the right-hand side of the equation, the package inverts the vertical velocity fields
at each time and depth in wavenumber space, i.e.

.. math::

    -\kappa^2N^2\hat{w} + f_0^2\frac{\partial^2 \hat{w}}{\partial z^2} = ik\beta\hat{b} + (ik\hat{Q}_x + il\hat{Q}_y).

where :math:`\mathbf{\kappa} = (k,l)` is the horizontal wavenumber vector.
The right-hand side, neglecting the turbulent correlation terms is
:math:`{\bf Q} = {\bf Q}_\text{tw} + {\bf Q}_\text{da}` where

.. math::

    {\bf Q}_\text{tw} = -2\Big(\frac{\partial {\bf u}}{\partial x} \cdot \nabla b, \frac{\partial {\bf u}}{\partial y} \cdot \nabla b\Big)

.. math::

    {\bf Q}_\text{da} = f \Big( \frac{\partial v}{\partial x}\frac{\partial u_\text{a}}{\partial z} - \frac{\partial u}{\partial x}\frac{\partial v_\text{a}}{\partial z}, \frac{\partial v}{\partial y}\frac{\partial u_\text{a}}{\partial z} - \frac{\partial u}{\partial y}\frac{\partial v_\text{a}}{\partial z} \Big).

Assuming the total flow to be in geostrophic balance :math:`{\bf u}={\bf u}_\text{g}=\frac{\hat{z}}{f}\times\nabla_\text{h}\Phi` reduces to the `quasi-geostrophic Omega equation <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.49710443903>`_.

.. note::

    xomega is at early stage of development and will keep improving in the future.
    The discrete Fourier transform API should be quite stable,
    but minor utilities could change in the next version.
    If you find any bugs or would like to request any enhancements,
    please `raise an issue on GitHub <https://github.com/roxyboy/xomega.git>`_.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Overview

   limitations

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 1
   :caption: API

   api


.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.pydata.org/en/latest/array-api.html
