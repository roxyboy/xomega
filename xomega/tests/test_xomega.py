import numpy as np
import xarray as xr
import numpy.testing as npt
import pytest
from scipy.interpolate import interp1d
from xomega import w_ageo

# @pytest.fixture(params=['numpy', 'xarray'])
def test_4D():
    """Check whether depth is decreasing monotonically."""
    N = 10
    da = np.random.rand(N,N,N,N)
    da = xr.DataArray(da, dims=['T','Zl','Y','X'],
                     coords={'T':range(N),'Zl':range(0,-10,-1),
                            'Y':range(N),'X':range(N)}
                     )
    Z = xr.DataArray(np.arange(-.5,-10.5,-1.), dims=['Z'],
                    coords={'Z':np.arange(-.5,-10.5,-1.)}
                    )
    dz = Z.diff('Z')
    f = interp1d(da.Zl[1:],dz,fill_value='extrapolate')
    dz = f(Z)
    DZ = da.Zl.diff('Zl')
    f = interp1d(Z[1:].data,DZ,fill_value='extrapolate')
    DZ = f(Z.data)

    with pytest.raises(NotImplementedError):
        w_ageo(da,da.Zl,0.,0.,0.,dz,DZ=DZ)

def test_dims():
    N = 10
    da = np.random.rand(N,N,N)
    da = xr.DataArray(da, dims=['Zl','Y','X'],
                     coords={'Zl':range(0,-10,-1),
                            'Y':range(N),'X':range(N)}
                     )
    Z = xr.DataArray(np.arange(-.5,-10.5,-1.), dims=['Z'],
                    coords={'Z':np.arange(-.5,-10.5,-1.)}
                    )
    dz = Z.diff('Z')
    f = interp1d(da.Zl[1:],dz,fill_value='extrapolate')
    dz = f(Z)
    DZ = da.Zl.diff('Zl')
    f = interp1d(Z[1:].data,DZ,fill_value='extrapolate')
    DZ = f(Z.data)

    with pytest.raises(ValueError):
        w_ageo(da,da.Zl,0,0,np.ones(N),dz,DZ=DZ,FTdim=['Y','X'])
    with pytest.raises(ValueError):
        w_ageo(da,da.Zl,0,0,np.ones(N-1),dz,FTdim=['Y','X'])

    # with pytest.raises(ValueError):
    #     xomega.w_ageo(da, Zl, dz, DZ, 0., 0., 0.)

    # ds.Zl[0] = -1.
    #
    # ds.coords['']
    # with pytest.raises(ValueError):
    #     xo.wa(ds, ds.psi, 0., 0., 0., grid='blah')
