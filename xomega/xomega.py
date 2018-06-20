import numpy as np
import xarray as xr
import dask.array as dsar
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.interpolate import PchipInterpolator as pchip
import xrft
import warnings

__all__ = ['w_ageo']

def w_ageo(N2, f0, beta, Frhs, dZ, DZ=None, zdim='Zl',
    kdim='freq_XC', ldim='freq_YC', dim=None, coord=None):
    """
    Inverts the quasi-geostrophic Omega equation to get the
    ageostrophic vertical velocity ($w_a$).

    .. math::

        Frhs = \beta\frac{\partial b}{\partial x} - 2 \nabla_z \cdot {\bf Q}
    where
        {\bf Q}\equiv (Q_1,Q_2) = \big(\frac{\partial{\bf u}_g}{\partial x}\cdot\nabla_zb,
                                       \frac{\partial{\bf u}_g}{\partial y}\cdot\nabla_zb
                                  \big).

    Parameters
    ----------
    N2 : `float` or `xarray.DataArray`
        The buoyancy frequency.
    f0 : `float`
        Coriolis parameter.
    beta : `float`
        Meridional gradient of the Coriolis parameter
        for a beta-plane approximation.
    Frhs : `xarray.DataArray`
        The Fourier transform of the right-hand side of
        the Omega equation.
    dZ : `float` or `xarray.DataArray`
        Vertical distance between grid mid points.
    DZ : `xarray.DataArray` (optional)
        This should only be specified when `dZ` is an array
        and the data is on non-uniform vertical grids.
    zdim : `str`
        Dimension name of the vertical axis of `Frhs`.
    dim : `list`
        List of the xarray.DataArray output.
    coord : `dict`
        Dictionary of the xarray.DataArray output.

    Returns
    -------
    wa : `xarray.DataArray`
        The quasi-geostrophic vertical velocity.
    """
    Zl = Frhs[zdim]
    kx = 2*np.pi*Frhs[kdim]
    ky = 2*np.pi*Frhs[ldim]
    N = Frhs.shape
    nz = N[0]
    if isinstance(N2, float):
        enu = eye(nz) * N2
    else:
        row = range(nz)
        col = range(nz)
        # if len(N2) != nz-1:
        #     raise ValueError("N2 should have one element less than psi.")
        if N2.dims != Zl.dims:
            raise ValueError("`N2` and `Frhs` should be on "
                            "the same vertical grid.")
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
    col = np.append(range(2), col)
    col = np.append(col, np.array([nz-2,nz-1]))

    if isinstance(dZ, float):
        dZ = dZ*np.ones(nz)
        DZ = dZ
    else:
        warnings.warn("The numerical errors for vertical derivatives "
                     "may be significant if the data is on a non-uniform "
                     "vertical grid.")
        if DZ == None:
            raise ValueError("If dz is an array, "
                            "DZ also needs to be an array.")
    tmp = np.zeros((nz-2,3))
    tmp[:,0] = DZ[:-2]**-1
    tmp[:,-1] = DZ[1:-1]**-1
    tmp[:,1] = -(DZ[:-2]**-1 + DZ[1:-1]**-1)
    data = np.zeros(len(tmp.ravel())+4)
    data[2:-2] = (tmp * dZ[1:nz-1,np.newaxis]**-1).ravel()
    data[0] = -dZ[0]**-1 * (DZ[0]**-1 + DZ[1]**-1)
    data[1] = dZ[0]**-1 * DZ[1]**-1
    data[-2] = dZ[nz-1]**-1 * DZ[-2]**-1
    data[-1] = -dZ[nz-1]**-1 * (DZ[-2]**-1 + DZ[-1]**-1)
    data *= f0**2
    delta = coo_matrix((data,(row,col)),
                      shape=(nz,nz),dtype=np.float64
                      )

    nk, nl = (len(kx),len(ky))
    wahat = np.zeros_like(Frhs, dtype=np.complex128)

    for i in range(nk):
        for j in range(nl):
            kappa2 = kx[i].data**2+ky[j].data**2
            ### Normal inversion ###
            A = csc_matrix(-kappa2*enu+delta, dtype=np.float64)
            ### Rigid lid solution ###
            wahat[:,j,i] = spsolve(A, Frhs[:,j,i])

    wahat = xr.DataArray(wahat, dims=[dim[0],kdims[-2],kdims[-1]],
                        coords={dim[0]:Zl.data,kdims[-2]:ky,kdims[-1]:kx}
                        )
    wa = dsar.fft.ifft2(wahat.chunk(chunks={dim[0]:1,
                                           kdims[-1]:N[-1],
                                           kdims[-2]:N[-2]}
                                   ).data, axes=[-2,-1]
                       ).real

    return xr.DataArray(wa, dims=dim, coords=coord)
