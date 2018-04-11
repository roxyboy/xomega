import numpy as np
import xarray as xr
import numpy.testing as npt
import pytest
from xomega import xomega

# @pytest.fixture(params=['numpy', 'xarray'])
def test_w_ageo():
    """Check whether depth is decreasing monotonically."""
    N = 10
    da = np.random.rand(N)
    da = xr.DataArray(da, dims=['Z'],
                     coords={'Z':np.arange(-.5,-10.5,-1.)})
    Zl = xr.DataArray(range(0,-10,-1), dims=['Zl'],
                      coords={'Zl':np.arange(0.,-10.,-1.)})
    dz = da.Z.diff('Z')
    DZ = Zl.diff('Zl')

    with pytest.raises(ValueError):
        xomega.w_ageo(da, Zl, dz, DZ, 0., 0., 0.)

    # ds.Zl[0] = -1.
    #
    # ds.coords['']
    # with pytest.raises(ValueError):
    #     xo.wa(ds, ds.psi, 0., 0., 0., grid='blah')
