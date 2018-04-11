import numpy as np
import xarray as xr
import dask.array as dsar
from scipy.linalg import solve
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import spsolve
import xrft
# from xgcm.grid import Grid
import warnings

__all__ = ['w_ageo']

def w_ageo(psi, Zl, dz, DZ, f0, beta, N2, grid=None, dim=None, coord=None):
    """
    Inverts the QG Omega equation to get the
    first-order ageostrophic vertical velocity.

    Parameters
    ----------
    psi  : xarray.DataArray
        Geostrophic stream function
    Zl   : xarray.DataArray
        Depth of top cell surface
    dz   : xarray.DataArray
        Difference between cell mid points.
    DZ   : xarray.DataArray
        Difference between cell interfaces.
    f0   : int
        Coriolis parameter
    beta : int
        Meridional gradient of the Coriolis parameter
    N2   : int or xarray.DataArray
        Buoyancy frequency
    grid : xgcm.grid object
        It will not work without grid specified.

    Returns
    -------
    wa   : xarray.DataArray
        Ageostrophic vertical velocity
    """
    if grid == None:
        raise ValueError("xgcm.Grid object needs to be provided.")

    # Z = ds.Z         # Depth between interface
    # Zp1 = ds.Zp1     # Depth of interface
    # Zl = ds.Zl       # Depth of top interface
    # DZ = ds.drF      # Difference between interface
    # dZ = ds.drC      # Difference between grid mid points
    Z = len(psi.Z)
    nz = len(Z)
    N = psi.shape

    if any(np.abs(Zl) > np.abs(Z)):
        raise ValueError("The top cell surface should be shallower"
                        "than the mid points.")

    # if any(DZ <= 0.) or any(dZ <= 0.):
    #     raise ValueError('Difference between depths should be'
    #                     'positive values.')
    # if any(np.abs(Zl[0])-np.abs(Z[0]) >= 0.):
    #     raise ValueError('Top of the grid interface should be'
    #                     'shallower than the mid points.')

    psihat = xrft.dft(psi.chunk(chunks={'Z':1}), dim=['Y','X'], shift=False
                     ).chunk(chunks={'freq_Y':N[-1],'freq_X':N[-2]})
    bhat = grid.interp(grid.diff(psihat,'Z',boundary='fill')
                      / grid.diff(psihat.Z,'Z',boundary='fill'),
                      'Z',boundary='fill')
    bhat *= f0
    kx = 2*np.pi*psihat.freq_X
    ky = 2*np.pi*psihat.freq_Y

    ughat = -1j*psihat*ky
    vghat = 1j*psihat*kx

    Q1 = (dsar.fft.ifft2(1j*(ughat*kx).data, axes=[-2,-1])
         * dsar.fft.ifft2(1j*(bhat*kx).data, axes=[-2,-1])
         + dsar.fft.ifft2(1j*(vghat*kx).data, axes=[-2,-1])
         * dsar.fft.ifft2(1j*(bhat*ky).data, axes=[-2,-1])
         )
    Q2 = (dsar.fft.ifft2(1j*(ughat*ky).data, axes=[-2,-1])
         * dsar.fft.ifft2(1j*(bhat*kx).data, axes=[-2,-1])
         + dsar.fft.ifft2(1j*(vghat*ky).data, axes=[-2,-1])
         * dsar.fft.ifft2(1j*(bhat*ky).data, axes=[-2,-1])
         )

    Q1hat = xrft.dft(Q1, dim=['Y','X'], shift=False)
    Q2hat = xrft.dft(Q2, dim=['Y','X'], shift=False)

    Frhs = (1j*beta*bhat*kx - 2*(1j*Q1hat*kx + 1j*Q2hat*ky)).compute()

    ### N2 matrix ###
    if type(N2).__name__ == 'int':
        enu = eye(nz) * N2
    else:
        row = range(nz)
        col = range(nz)
        enu = coo_matrix((N2,(row,col)),
                        shape=(nz,nz), dtype=np.float64
                        )

    ### Delta matrix ###
    row = np.repeat(range(1,nz-1),3)
    row = np.append(np.array([0,0]),row)
    row = np.append(row,np.array([nz-1,nz-1]))
    col = np.arange(3)
    for i in range(nz-3):
        col = np.append(col, np.arange(i+1,i+4))
    col = np.append(range(2),col)
    col = np.append(col,np.array([nz-2,nz-1]))

    tmp = np.zeros((nz-2,3))
    tmp[:,0] = DZ[:-2].data**-1
    tmp[:,-1] = DZ[1:-1].data**-1
    tmp[:,1] = -(DZ[:-2].data**-1 + DZ[1:-1].data**-1)
    data = np.zeros(len(tmp.ravel())+4)
    data[2:-2] = (tmp/dZ[:-1,np.newaxis]).ravel()
    data[0] = -dZ[0].data**-1 * DZ[0].data**-1
    data[1] = dZ[0].data**-1 * DZ[0].data**-1
    data[-2] = dZ[-2].data * DZ[-2].data**-1
    data[-1] = -dZ[-2].data * (DZ[-2].data**-1 + DZ[-1].data**-1)
    # data[-1] = -rho_mid[nz]/dZ[-1] * (rho[-1]*DZ[-1])**-1
    data *= f0**2
    delta = coo_matrix((data,(row,col)),
                      shape=(nz,nz),dtype=np.float64
                      )

    nk, nl = (len(kx),len(ky))
    wahat = np.zeros((nz,nl,nk), dtype=np.complex128)

    for i in range(nk):
        for j in range(nl):
            kappa2 = kx[i].data**2+ky[j].data**2
            ### Normal inversion ###
            A = csc_matrix(-kappa2*enu+delta, dtype=np.float64)
            wahat[:,j,i] = spsolve(A, Frhs[:,j,i])

    wahat = xr.DataArray(wahat, dims=['Zl','freq_Y','freq_X'],
                        coords={'Zl':Zl.data,'freq_y':ky,'freq_X':kx}
                        )
    wa = dsar.fft.ifft2(wahat.chunk(chunks={'Z':1,'freq_Y':N[-1],'freq_X':N[-2]}
                                   ).data, axes=[-2,-1]
                       ).real

    return xr.DataArray(wa, dims=dim, coords=coord)
