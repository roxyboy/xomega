import numpy as np
import xarray as xr
import numpy.testing as npt
import pytest
import os.path as op
from scipy.io import loadmat
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
        w_ageo(da,0.,0.,0.,dz,DZ=DZ)

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
        w_ageo(da.chunk(chunks={'Zl':1}),0,0,
              xr.DataArray(np.ones(N+1),dims=['Zp1'],
                          coords={'Zp1':range(0,-11,-1)}),
              dz,DZ=DZ,FTdim=['Y','X'])
    with pytest.raises(ValueError):
        w_ageo(da.chunk(chunks={'Zl':1}),0,0,
              xr.DataArray(np.ones(N),dims=da.Zl.dims,
                          coords={'Zl':da.Zl.data}),
              dz,FTdim=['Y','X'])
    with pytest.raises(ValueError):
        w_ageo(da.chunk(chunks={'Zl':1}),0,0,
              xr.DataArray(np.ones(N),dims=da.Zl.dims,
                          coords={'Zl':da.Zl.data}),
              dz,DZ=DZ,FTdim=['Y','X'],periodic='X')

# def test_qg():
#     TESTDATA_FILENAME = op.join(op.dirname(__file__),
#                                'QG_psi-and-w.nc')
#     ds = xr.open_dataset(TESTDATA_FILENAME)
#
#     dz = np.abs(ds.Z.diff('Z')[0].data)
#     psi = xr.DataArray(.5*(ds.psi_uni+ds.psi_uni.shift(Z=-1))[:-1].data,
#                       dims=['Zb','Y','X'],
#                       coords={'Zb':ds.Zb.data,'Y':ds.Y.data,'X':ds.X.data}
#                       )
#     wa = w_ageo(psi.chunk(chunks={'Zb':1}), 0.00010131036606448109,
#                1.6448722979145434e-11, 4.009293075046547e-07, dz,
#                zdim='Zb', FTdim=['Y','X'])
#
#     npt.assert_allclose((wa**2).mean(['X','Y']),
#                        (ds.w**2).mean(['X','Y']), rtol=1e-1)

    # with pytest.raises(ValueError):
    #     xomega.w_ageo(da, Zl, dz, DZ, 0., 0., 0.)

    # ds.Zl[0] = -1.
    #
    # ds.coords['']
    # with pytest.raises(ValueError):
    #     xo.wa(ds, ds.psi, 0., 0., 0., grid='blah')
