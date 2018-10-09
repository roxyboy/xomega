import numpy as np
import xarray as xr
import dask.array as dsar
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.interpolate import PchipInterpolator as pchip
# import xrft
import warnings

__all__ = ['w_ageo_rigid']

def w_ageo_rigid(N2, f0, beta, Frhs, kx, ky, dZ, dZ0=None, dZ1=None, zdim='Zl',
                dim=None, coord=None):
    """
    Inverts the Omega equation given by Giordani and Planton (2000)
    to get the ageostrophic vertical velocity ($w_a$)
    for rigid lid boundary conditions.

    .. math::
        \nabla_z^2 N^2 w_a + f_0^2 \frac{\partial^2 w_a}{\partial z^2} = \beta\frac{\partial b}{\partial x} + \nabla_z \cdot {\bf Q}

    Parameters
    ----------
    N2 : float or xarray.DataArray
        The buoyancy frequency.
    f0 : float
        Coriolis parameter.
    beta : float
        Meridional gradient of the Coriolis parameter
        for a beta-plane approximation.
    Frhs : xarray.DataArray
        The Fourier transform of the right-hand side of
        the Omega equation. The last two dimensions should be
        the meridional and zonal wavenumber.
    kx : xarray.DataArray
        Zonal wavenumber.
    ky : xarray.DataArray
        Meridional wavenumber.
    dZ : float or xarray.DataArray
        Vertical distance between grid.
    dZ0 : float or xarray.DataArray, optional
        Top vertical distance between grids.
    dZ1 : float or xarray.DataArray, optional
        Bottom vertical distance between grids.
    zdim : str, optional
        Dimension name of the vertical axis of `Frhs`.
    dim : list, optional
        List of the xarray.DataArray output.
    coord : dict, optional
        Dictionary of the xarray.DataArray output.
    wvnm : bool, optional
        Whether the coordinates of `Frhs` are wavenumbers
        or inverse wavelengths. Default is `False` meaning
        that the coordinates are in the latter.

    Returns
    -------
    wa : xarray.DataArray
        The quasi-geostrophic vertical velocity.
    """
    Zl = Frhs[zdim]
    # kdims = Frhs.dims[-2:]
    N = Frhs.shape
    nz = N[0]

    if isinstance(N2, float):
        enu = eye(nz-2) * N2
    else:
        if nz-2 != len(N2):
            raise ValueError("N2 should have two elements less than Frhs.")
        row = range(nz-2)
        col = range(nz-2)
        # if len(N2) != nz-1:
        #     raise ValueError("N2 should have one element less than psi.")
        if N2.dims != Zl.dims:
            raise ValueError("N2 and psi should be on the same vertical grid.")
        enu = coo_matrix((N2,(row,col)),
                        shape=(nz,nz), dtype=np.float64
                        )

    ### Delta matrix ###
    row = np.repeat(range(1,nz-1),3)
    row = np.append(np.array([0]), row)
#     row = np.append(0, row)
    row = np.append(row, np.array([nz-1]))
#     row = np.append(row, nz)
    col = np.arange(3)
    for i in range(nz-3):
        col = np.append(col, np.arange(i+1,i+4))
    col = np.append(0, col)
#     col = np.append(0, col)
    col = np.append(col, np.array([nz-1]))
#     col = np.append(col, nz-1)
    if dZ0 == None:
        dZ0 = dZ
    if dZ1 == None:
        dZ1 = dZ
    if isinstance(dZ, float):
        dZ = dZ*np.ones(nz)
        DZ = dZ
        dZ = np.append(dZ0, dZ)
        dZ = np.append(dZ, dZ1)
    else:
        warnings.warn("The data is not on uniform vertical gridding. "
                     "The numerical errors for vertical derivatives "
                     "may be significant.")
    tmp = np.zeros((nz-2,3))
    tmp[:,0] = DZ[:-2]**-1
    tmp[:,-1] = DZ[1:-1]**-1
    tmp[:,1] = -(DZ[:-2]**-1 + DZ[1:-1]**-1)
    data = np.zeros(len(tmp.ravel())+2)
    data[1:-1] = (tmp * dZ[1:nz-1,np.newaxis]**-1).ravel()
#     data[0] = dZ[0]**-1 * DZ[0]
    data[0] = 1
    data[-1] = 1
#     data[-1] = dZ[-1]**-1 * DZ[-1]**-1
    data *= f0**2
    delta = coo_matrix((data,(row,col)),
                      shape=(nz,nz),dtype=np.float64
                      )

    # ky = Frhs[kdims[0]]
    # kx = Frhs[kdims[1]]
    # if wvnm == False:
    #     warnings.warn("The coordinates are in inverse wavelenths so "
    #                  "converting them to wavenumbers.")
    #     ky *= 2*np.pi
    #     kx *= 2*np.pi
    nk, nl = (len(kx),len(ky))
    wahat = np.zeros_like(Frhs, dtype=np.complex128)

    for i in range(nk):
        for j in range(nl):
            kappa2 = kx[i].data**2+ky[j].data**2
            ### Normal inversion ###
#             A = csc_matrix(delta, dtype=np.float64)
#             A[1:-1] -= csc_matrix(kappa2*enu, dtype=np.float64)
            A = csc_matrix(-kappa2*enu+delta, dtype=np.float64)
            ### Rigid lid solution ###
            wahat[:,j,i] = spsolve(A, Frhs[:,j,i])
#             wahat[1:-1,j,i], res, rnk, s = lstsq(A.todense(), Frhs[:,j,i])

    wahat = xr.DataArray(wahat, dims=[dim[0],kdims[-2],kdims[-1]],
                        coords={dim[0]:Zl.data,kdims[-2]:ky,kdims[-1]:kx}
                        )
    wa = dsar.fft.ifft2(wahat.chunk(chunks={dim[0]:1,
                                           kdims[-1]:N[-1],
                                           kdims[-2]:N[-2]}
                                   ).data, axes=[-2,-1]
                       ).real

    return xr.DataArray(wa, dims=dim, coords=coord)
