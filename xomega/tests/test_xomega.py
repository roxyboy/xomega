import numpy as np
import xarray as xr
import numpy.testing as npt
import pytest
import xomega

@pytest.fixture(params=['numpy', 'xarray'])
def test_wa():
    """Check whether the depth is increasing monotonically."""
    N = 10
    da = np.random.rand(N)
    da = xr.DataArray(da, dims=['Z'], coords={'Z':np.arange(.5,10.5,1.)})
    ds = da.to_dataset(name='psi')
    ds.coords['Zp1'] = ('Zpl',range(11))
    ds.coords['Zl'] = ('Zl',range(10))
    ds.Zl[0] = 1.

    with pytest.raises(ValueError):
        wa(ds, ds.psi, 0., 0., 0.)
