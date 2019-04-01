[![Build Status](https://travis-ci.org/roxyboy/xomega.svg?branch=master)](https://travis-ci.org/roxyboy/xomega)
[![DOI](https://zenodo.org/badge/125751493.svg)](https://zenodo.org/badge/latestdoi/125751493)

# xomega: Inversion of the QG Omega equation
Functions to invert the [generalized Omega equation](https://journals.ametsoc.org/doi/full/10.1175/1520-0493%282000%29128%3C2270%3AMAAOAC%3E2.0.CO%3B2),
developed by [Takaya Uchida]( https://roxyboy.github.io/ )
at Columbia University in the City of New York.

It solves the following equation
$$N^2\nabla_\text{h}w_\text{b} + f_0^2\frac{\partial^2w_\text{b}}{\partial z^2} = \beta\frac{\partial b}{\partial x} + \nabla_\text{h}\cdot{\bf Q}(u,v,\theta,\Phi)$$ 
in wavenumber space at each time and depth level
$$-\kappa^2N^2\hat{w} + f_0^2\frac{\partial^2 \hat{w}}{\partial z^2} = ik\beta\hat{b} + (ik\hat{Q}_x + il\hat{Q}_y).$$
where $\mathbf{\kappa} = (k,l)$ is the horizontal wavenumber vector.
The right-hand side, neglecting the turbulent correlation terms is
${\bf Q} = {\bf Q}_\text{tw} + {\bf Q}_\text{da}$ where
$${\bf Q}_\text{tw} &= -2\Big(\frac{\partial {\bf u}}{\partial x} \cdot \nabla b, \frac{\partial {\bf u}}{\partial y} \cdot \nabla b\Big) \label{eqn:qtw}$$
$${\bf Q}_\text{da} &= f \Big( \frac{\partial v}{\partial x}\frac{\partial u_\text{a}}{\partial z} - \frac{\partial u}{\partial x}\frac{\partial v_\text{a}}{\partial z}, \frac{\partial v}{\partial y}\frac{\partial u_\text{a}}{\partial z} - \frac{\partial u}{\partial y}\frac{\partial v_\text{a}}{\partial z} \Big).$$

Assuming the total flow to be in geostrophic balance (${\bf u}={\bf u}_\text{g}=\frac{\hat{z}}{f}\times\nabla_\text{h}\Phi$) reduces eqn.~(\ref{eqn:omega}) to the [quasi-geostrophic Omega equation](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.49710443903). The ageostrophic velocities were defined as the difference between the total and geostrophic velocity, i.e.~${\bf u}_\text{a} = {\bf u} - {\bf u}_\text{g}$.
