import numpy as np
import xarray as xr
import dask.array as dsar
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import xrft
import warnings

__all__ = ['w_ageo']

def w_ageo(psi, f0, beta, N2, dz, DZ=None, zdim='Zl',
          grid=None, FTdim=None, dim=None, coord=None):
    """
    Inverts the QG Omega equation to get the
    first-order ageostrophic vertical velocity.

    Parameters
    ----------
    psi  : dask.array
        Geostrophic stream function. It should be aligned
        on the cell top. The 2D FFT will be taken so the
        horizontal axes should not be chunked.
    f0   : float
        Coriolis parameter.
    beta : float
        Meridional gradient of the Coriolis parameter.
    N2   : float or xarray.DataArray
        Buoyancy frequencys squared.
    dz   : float or numpy.array
        Difference between cell mid points.
    DZ   : numpy.array (conditional)
        Difference between cell interfaces.
    zdim : str
        Name of the vertical dimension of psi.
    grid : xgcm.grid object (optional)
        Uses the xgcm.grid.Grid functionality to take
        the differencing and interpolation.

    Returns
    -------
    wa   : xarray.DataArray
        Ageostrophic vertical velocity
    """
    # if grid == None:
    #     raise ValueError("xgcm.Grid object needs to be provided.")

    Zl = psi[zdim]
    nz = len(Zl)
    N = psi.shape
    if len(N) != 3:
        raise NotImplementedError("Taking data with more than 3 dimensions "
                                 "is not implemented yet.")

    if dim == None:
        dim = psi.dims
    if coord == None:
        coord = psi.coords

    psihat = xrft.dft(psi, dim=FTdim, shift=False)
    if grid == None:
        bhat = psihat.diff(zdim)/Zl.diff(zdim)
        axis_num = psi.get_axis_num(zdim)
        func = interp1d(.5*(Zl[1:].data+Zl[:-1].data), bhat,
                       axis=axis_num, fill_value='extrapolate'
                       )
        bhat = xr.DataArray(func(Zl.data), dims=psihat.dims,
                           coords=psihat.coords
                           ).chunk(chunks=psihat.chunks)
    else:
        bhat = grid.interp(grid.diff(psihat,'Z',boundary='fill')
                          / grid.diff(Zl,'Z',boundary='fill'),
                          'Z', boundary='fill'
                          ).chunk(chunks=psihat.chunks)
        if psihat.dims != bhat.dims:
            raise ValueError("psihat and bhat should be on the same grid.")
    bhat *= f0

    # k_names = ['freq_' + d for d in psihat.dims[-2:]]
    kx = 2*np.pi*psihat[psihat.dims[-1]]
    ky = 2*np.pi*psihat[psihat.dims[-2]]

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

    Q1hat = xrft.dft(xr.DataArray(Q1,dims=psi.dims,coords=psi.coords),
                    dim=FTdim, shift=False)
    Q2hat = xrft.dft(xr.DataArray(Q2,dims=psi.dims,coords=psi.coords),
                    dim=FTdim, shift=False)

    Frhs = (1j*beta*bhat*kx - 2*(1j*Q1hat*kx + 1j*Q2hat*ky)).compute()

    ### N2 matrix ###
    if isinstance(N2, float):
        enu = eye(nz-1) * N2
    else:
        row = range(nz-1)
        col = range(nz-1)
        # if len(N2) != nz-1:
        #     raise ValueError("N2 should have one element less than psi.")
        if N2.dims != Zl.dims:
            raise ValueError("N2 and psi should be on the same vertical grid.")
        enu = coo_matrix((N2[1:],(row,col)),
                        shape=(nz-1,nz-1), dtype=np.float64
                        )

    ### Delta matrix ###
    row = np.repeat(range(1,nz-2),3)
    row = np.append(np.array([0,0]),row)
    row = np.append(row,np.array([nz-2,nz-2]))
    col = np.arange(3)
    for i in range(nz-4):
        col = np.append(col, np.arange(i+1,i+4))
    col = np.append(range(2), col)
    col = np.append(col, np.array([nz-3,nz-2]))

    if isinstance(dz, float):
        dz = dz*np.ones(nz)
        DZ = dz
    else:
        warnings.warn("The numerical errors for vertical derivatives "
                     "may be significant.")
        if DZ == None:
            raise ValueError("If dz is an array, "
                            "DZ also needs to be an array.")
    tmp = np.zeros((nz-3,3))
    tmp[:,0] = DZ[1:-2]**-1
    tmp[:,-1] = DZ[2:-1]**-1
    tmp[:,1] = -(DZ[1:-2]**-1 + DZ[2:-1]**-1)
    data = np.zeros(len(tmp.ravel())+4)
    data[2:-2] = (tmp * dZ[2:nz-1,np.newaxis]**-1).ravel()
    data[0] = -dZ[1]**-1 * (DZ[0]**-1 + DZ[1]**-1)
    data[1] = dZ[1]**-1 * DZ[1]**-1
    data[-2] = dZ[nz-1]**-1 * DZ[-2]**-1
    data[-1] = -dZ[nz-1]**-1 * (DZ[-2]**-1 + DZ[-1]**-1)
    data *= f0**2
    delta = coo_matrix((data,(row,col)),
                      shape=(nz-1,nz-1),dtype=np.float64
                      )

    nk, nl = (len(kx),len(ky))
    wahat = np.zeros((nz+1,nl,nk), dtype=np.complex128)

    for i in range(nk):
        for j in range(nl):
            kappa2 = kx[i].data**2+ky[j].data**2
            ### Normal inversion ###
            A = csc_matrix(-kappa2*enu+delta, dtype=np.float64)
            ### Rigid lid solution ###
            wahat[1:-1,j,i] = spsolve(A, Frhs[1:,j,i])

    wahat = xr.DataArray(wahat, dims=['Zl','freq_Y','freq_X'],
                        coords={'Zl':Zl.data,'freq_y':ky,'freq_X':kx}
                        )
    wa = dsar.fft.ifft2(wahat.chunk(chunks={'Z':1,'freq_Y':N[-1],'freq_X':N[-2]}
                                   ).data, axes=[-2,-1]
                       ).real

    return xr.DataArray(wa, dims=dim, coords=coord)
