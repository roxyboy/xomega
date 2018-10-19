.. xomega documentation master file, created by
   sphinx-quickstart on Fri Oct 19 11:42:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xomega's documentation!
==================================
xomega is a Python package for solving the
`generalized omega equation <https://journals.ametsoc.org/doi/abs/10.1175/1520-0493(2000)128%3C2270:MAAOAC%3E2.0.CO%3B2>`_.
Given the right-hand side of the equation, the package inverts the vertical velocity fields.

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
