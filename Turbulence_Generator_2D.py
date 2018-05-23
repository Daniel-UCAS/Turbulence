#
#  IsoTurbGen.h
#
#  The MIT License (MIT)
#
#  Copyright (c) 2015, Tony Saad. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#

# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:31:54 2014

@author: tsaad
"""
import numpy as np
from numpy import sin, cos, sqrt, ones, zeros, pi, arange, conj, convolve
import matplotlib.pyplot as plt
from numpy.fft import fftn

#------------------------------------------------------------------------------

def movingaverage(interval, window_size):
    window= ones(int(window_size))/float(window_size)
    return convolve(interval, window, 'same')

#------------------------------------------------------------------------------

def compute_tke_spectrum_1d(u, lx, ly, lz, smooth):
    """
    Given a velocity field u this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the
    following steps:
    1. Compute the spectral representation of u using a fast Fourier transform.
    This returns uf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf)* conjugate(uf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).
    Parameters:
    -----------
    u: 3D array
      The x-velocity component.
    v: 3D array
      The y-velocity component.
    w: 3D array
      The z-velocity component.
    lx: float
      The domain size in the x-direction.
    ly: float
      The domain size in the y-direction.
    lz: float
      The domain size in the z-direction.
    smooth: boolean
      A boolean to smooth the computed spectrum for nice visualization.
    """
    nx = len(u[:, 0, 0])
    ny = len(u[0, :, 0])
    nz = len(u[0, 0, :])

    nt = nx * ny * nz
    n = max(nx, ny, nz)  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(u) / nt

    # tkeh = zeros((nx, ny, nz))
    tkeh = 0.5 * (uh * conj(uh)).real

    length = max(lx, ly, lz)

    knorm = 2.0 * pi / length

    kxmax = nx / 2
    kymax = ny / 2
    kzmax = nz / 2

    wave_numbers = knorm * arange(0, n)
    tke_spectrum = zeros(len(wave_numbers))

    for kx in range(nx):
        rkx = kx
        if kx > kxmax:
            rkx = rkx - nx
        for ky in range(ny):
            rky = ky
            if ky > kymax:
                rky = rky - ny
            for kz in range(nz):
                rkz = kz
                if kz > kzmax:
                    rkz = rkz - nz
                rk = sqrt(rkx * rkx + rky * rky + rkz * rkz)
                k = int(np.round(rk))
                #print('k = ', k)
                tke_spectrum[k] = tke_spectrum[k] + tkeh[kx, ky, kz]

    tke_spectrum = tke_spectrum / knorm

    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm * min(nx, ny, nz) / 2

    return knyquist, wave_numbers, tke_spectrum


#------------------------------------------------------------------------------

def passot_pouquet_spectrum(k,l):
    up = 0.5
    l11 = l/3
    ke = np.sqrt(2*np.pi)/l11    
    print('Ke = %.4f' % ke)
    C = np.sqrt(2/np.pi)*np.power(up,2.0)/ke*32.0/3.0
    E = C*np.power(k/ke,4.0)*np.exp(-2.0*np.power(k/ke,2.0))
    return E

def generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,especf):
  """
  Given an energy spectrum, this function computes a discrete, staggered, three 
  dimensional velocity field in a box whose energy spectrum corresponds to the input energy 
  spectrum up to the Nyquist limit dictated by the grid

  This function returns u, v, w as the axial, transverse, and azimuthal velocities.
  
  Parameters:
  -----------  
  lx: float
    The domain size in the x-direction.
  ly: float
    The domain size in the y-direction.
  lz: float
    The domain size in the z-direction.  
  nx: integer
    The number of grid points in the x-direction.
  ny: integer
    The number of grid points in the y-direction.
  nz: integer
    The number of grid points in the z-direction.
  wn1: float
    Smallest wavenumber. Typically dictated by spectrum or domain size, 2*dx
  espec: functor
    A callback function representing the energy spectrum.
  """
    # generate cell centered x-grid
  dx = float(lx/nx)
  dy = float(ly/ny)  
  dz = float(lz/nz)
  
  ## START THE FUN!
  # compute random angles
  phi =   2.0*pi*np.random.uniform(0.0,1.0,nmodes);
  nu = np.random.uniform(0.0,1.0,nmodes);
  theta = np.arccos(2.0*nu -1.0);
  psi   = np.random.uniform(-pi/2.0,pi/2.0,nmodes);
  
  
  # highest wave number that can be represented on this grid (nyquist limit)
  wnn = max(np.pi/dx, np.pi/dy, np.pi/dz)
  print( 'I will generate data up to wave number: ', wnn)
  
  # wavenumber step
  dk = (wnn-wn1)/nmodes
  print(dk)
  # wavenumber at cell centers
  wn = wn1 + 0.5*dk + arange(0,nmodes)*dk

  dkn = ones(nmodes)*dk
 
 
  
  #   wavenumber vector from random angles
  kx = sin(theta)*cos(phi)*wn
  ky = sin(theta)*sin(phi)*wn
  kz = cos(theta)*wn

  # create divergence vector
  ktx = np.sin(kx*dx/2.0)/(dx)
  kty = np.sin(ky*dy/2.0)/(dy)
  ktz = np.sin(kz*dz/2.0)/(dz)    

  # Enforce Mass Conservation
  phi1 =   2.0*pi*np.random.uniform(0.0,1.0,nmodes);
  nu1 = np.random.uniform(0.0,1.0,nmodes);
  theta1 = np.arccos(2.0*nu1 -1.0);
  zetax = sin(theta1)*cos(phi1)
  zetay = sin(theta1)*sin(phi1)
  zetaz = cos(theta1)
  sxm =  zetay*ktz - zetaz*kty
  sym = -( zetax*ktz - zetaz*ktx  )
  szm = zetax*kty - zetay*ktx
  smag = sqrt(sxm*sxm + sym*sym + szm*szm)
  sxm = sxm/smag
  sym = sym/smag
  szm = szm/smag  
    
  # verify that the wave vector and sigma are perpendicular
  kk = np.sum(ktx*sxm + kty*sym + ktz*szm)
  print ('Orthogonality of k and sigma (divergence in wave space:', kk)
  
  # get the modes   
  km = wn
  #quit()
  espec = especf(km, lx)
  espec = espec.clip(0.0)
  E_ = espec
  
  # generate turbulence at cell centers
  um = sqrt(espec*dkn)
  u_ = zeros([nx,ny,nz])
  v_ = zeros([nx,ny,nz])
  w_ = zeros([nx,ny,nz])

  xc = dx/2.0 + arange(0,nx)*dx  
  yc = dy/2.0 + arange(0,ny)*dy  
  zc = dz/2.0 + arange(0,nz)*dz
   
  for k in range(0,nz):
    for j in range(0,ny):
      u_[0,j,k] = 1
      for i in range(0,nx):
        #for every grid point (i,j,k) do the fourier summation 
        arg = kx*xc[i] + ky*yc[j] + kz*zc[k] - psi
        bmx = 2.0*um*cos(arg - kx*dx/2.0)
        bmy = 2.0*um*cos(arg - ky*dy/2.0)        
        bmz = 2.0*um*cos(arg - kz*dz/2.0)                
        #u_[i,j,k] = 1+k#np.sum(bmx*sxm)
        #print(np.sum(bmx*sxm))
        #quit()
        v_[i,j,k] = np.sum(bmy*sym)
        w_[i,j,k] = np.sum(bmz*szm)
  
        
  print ('done. I am awesome!')
  #print(u_)
  #quit()
  return u_, v_, w_, E_, km

U, V, W, E, K = generate_isotropic_turbulence(1.13,0.565,0.565,64,32,32,1000,2*18e-6,passot_pouquet_spectrum)
##NQ, KQ, EQ = compute_tke_spectrum_1d(U,1.13,0.565,0.565,True)
##plt.clf()
##plt.plot(KQ, EQ)
##plt.show()
with open('U.bin', 'wb') as f:
    f.write(U)

#kn, k, Ek = compute_tke_spectrum_1d(U,21.582,0.565,0.565,True)
##kn, k, Ek = compute_tke_spectrum_1d(U,0.002682,0.002682,0.002682,'false')
##print(len(k), len(Ek))
#plt.clf()
#plt.plot(k,Ek,'blue')
##plt.plot(K,E,'green')
#plt.show()
