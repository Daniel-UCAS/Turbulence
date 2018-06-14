"""
    File name: Turbulence_3D_3Vectors.py
    Author: Marvin Joshi
    Date created: 06/07/2018
    Python Version: 3.6

When creating the BOV File that goes into visit, DATA_SIZE is written as nz, ny, nx
and brick size is written as lz, ly, lx
EXAMPLE:
DATA_FILE: Z.bin
DATA_SIZE: NZ	NY   NX
DATA_FORMAT: FLOAT
VARIABLE: X
BRICK_ORIGIN: +0.000 +0.000 +0.000
BRICK_SIZE:   +LZ +LY  +LX
"""
#------------------------------------------------------------------------------

import numpy as np
from numpy import sin, cos, sqrt, ones, zeros, pi, arange, conj, convolve
import matplotlib.pyplot as plt
from numpy.fft import fftn
import csv
#------------------------------------------------------------------------------
def passot_pouquet_spectrum(k,lmin):
    up = 0.5
    l11 = lmin/3
    ke = np.sqrt(2*np.pi)/l11    
    #print('Ke = %.4f' % ke)
    C = np.sqrt(2/np.pi)*np.power(up,2.0)/ke*32.0/3.0
    E = C*np.power(k/ke,4.0)*np.exp(-2.0*np.power(k/ke,2.0))
    return E, up, ke
#------------------------------------------------------------------------------

def turb_3d(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz):
    i_cmplx = np.complex(0,1) 
    delta = np.max(((xmax-xmin)/(nx-1)) and ((ymax-ymin)/(ny-1)) and ((zmax-zmin)/(nz-1)))
    kc = np.pi/delta
    uk = np.zeros([nx,ny,nz,3],dtype=complex)
    q1 = np.zeros([nx, ny, nz,3], dtype = complex)
    q = np.zeros([nx, ny, nz,3], dtype = complex)
    X_FFT = np.zeros([nx, ny, nz,3], dtype = complex)
    
    #determine the amplitude for the passot-poquet spectrum   
    amp = (sqrt((2.0*np.pi)**3)/((xmax-xmin) * (ymax-ymin) * (zmax-zmin)))
   
    
    for k in range(0,nz):
        #Sets the point, which is called the pi-wavenumber, to 0
        if ((2*int(nz/2) == nz) and (k == nz/2)):
            continue        
        for j in range(0,ny):
            #Sets the point, which is called the pi-wavenumber, to 0 
            if ((2*int(ny/2) == ny) and (j == ny/2)):
                continue
            for i in range(0,int(nx/2)):
                """create random angles
                    - ps1 and ps2 ranges from -pi to pi
                    - psr ranges from 0 to 2*pi
                """
                ps1   = np.random.uniform(-pi,pi)
                ps2   = np.random.uniform(-pi,pi)
                psr   = np.random.uniform(0,2.0*pi)
                #Leave the origin value as 0, otherwise a mean velocity will occur
                if ((i == 0) and (j == 0) and (k == 0)):
                    continue
                    #Sets the point, which is called the pi-wavenumber, to 0
                if ((2*int(nx/2) == nx) and (i == nx/2)):
                    continue
                kx = (i)*2.0*pi/(xmax-xmin)
                if (j <= ny/2):
                    ky=(j)*2.0*pi/(ymax-ymin)
                else:
                    ky = -1.0*(ny-j)*2.0*pi/(ymax-ymin)
                if (k <=  nz/2):
                    kz = (k)*2.0*pi/(zmax-zmin)
                else:
                    kz = -1.0*(nz-k)*2.0*pi/(zmax-zmin)
                
                k2 = sqrt(kx**2 + ky**2)
                kmag = sqrt(kx**2 + ky**2 + kz**2)
                #Checks to see that the wave numvers are not larger than the cutoff
                if (kmag > kc):
                    continue
                e_spec, up, ke = passot_pouquet_spectrum(kmag, min(xmax, ymax, zmax))
                ak = amp*sqrt(e_spec/(2.0*pi*kmag**2))*np.exp(i_cmplx*ps1)*cos(psr)
                bk = amp*sqrt(e_spec/(2.0*pi*kmag**2))*np.exp(i_cmplx*ps2)*sin(psr)
                #Calculates the turbulence values
                if(k2 < (ke/1e4)):
                    uk[i, j, k,0] = (ak+bk)/sqrt(2.0)
                else:
                    uk[i, j, k,0] = (ak*kmag*ky+bk*kx*kz)/(kmag*k2)
                if(k2 < (np.sqrt(2*np.pi)/(xmax/3))/1e4): 
                    uk[i, j, k, 1] = (bk-ak)/sqrt(2.0)
                else:
                    uk[i, j, k, 1] = (bk*ky*kz-ak*kmag*kx)/(kmag*k2)
                uk[i, j, k, 2] = -(bk*k2)/kmag


    conjugate_yzplane(uk[0,:,:,0],nx,ny,nz)
    conjugate_yzplane(uk[0,:,:,1],nx,ny,nz)
    conjugate_yzplane(uk[0,:,:,2],nx,ny,nz) 
     

    #Finds the FFT in the Y-Direction
    Y_FFT = np.fft.fft(uk, axis=1)
    
    #Finds the FFT in the Z-Direction
    Z_FFT = np.fft.fft(Y_FFT,axis=2)
    
    #Finds the FFT in the X-Direction after performing the conjugate
    for k in range(0,nz):
        for j in range(0,ny):
            q[0:nx,j,k] = Z_FFT[0:nx,j,k]
            for i in range(int(1+nx/2),nx):
                q[i,j,k] = conj(q[nx-i,j,k])
            q1 = np.fft.fft(q,axis = 0)
            X_FFT[0:nx,j,k] = q1[0:nx,j,k]
    
    """ To verify that the turbulence is generated correctly, the data must satisfy these three conditions:
    1. The sum of the real values after the FFTs should be equal or close to 0.
    2. The maximum imaginary value after the FFTs should be equal or close to 0.
    3. The Spectral Energy Content and the Physical Energy Content should be the sane. 
        - To determine the Spectral Energy Content, you use the turbuelence array before the FFTs.
        - TO determine the Physical Energy Content, you use the turbuelence array after the FFTs.
    """
    #Sum of Real Values
    sum_real = np.sum(X_FFT.real)
    
    #Maximum Imaginary Value
    max_imag = np.amax(np.abs(np.imag(X_FFT)))
    
    #Spectral Energy Content
    spectral_energy = np.sum(np.real(uk*np.conj(uk)))
    spectral_energy = spectral_energy - 0.5*np.sum(np.real(uk[0,:,:]*np.conj(uk[0,:,:])))
    spectral_energy = spectral_energy/(1.5*up**2)

    #Physical Energy Content
    physical_energy = 0.5* np.sum(X_FFT.real**2)
    physical_energy = physical_energy/(1.5*up**2*nx*ny*nz)

    print("Sum of the Real Values: ", sum_real)
    print("Maximum Imaginary Value: ", max_imag)
    print("Spectral Energy Content: ", spectral_energy)
    print("Physical Energy Content: ", physical_energy)
   
    U = X_FFT[:,:,:,0].real
    V = X_FFT[:,:,:,1].real
    W = X_FFT[:,:,:,2].real
    mag_vector = np.sqrt(U**2 + V**2 + W**2)
    y = np.ascontiguousarray(mag_vector, dtype=np.float32)
    with open('Z.bin', 'wb') as f:
        f.write(y)

    """
    csv_file = '3D_FFT_Output.csv'
    
    with open(csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['X','Y','Z','Z_real','Z-imag'])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    writer.writerow([i,j,k,z2.real[i,j,k],z2.imag[i,j,k]])
    print('\n Output written to file')
    """

def conjugate_yzplane(f,nx,ny,nz):
    for k in range(int(1+nz/2),nz):
        #Finds the conjugate values across the y = 0 plane
        f[0,k] = conj(f[0,nz-k])
        for j in range(1,ny):
            #Finds the conjugate values for every y and z points
            f[j,k] = conj(f[ny-j,nz-k])
    for j in range(int(1+ny/2),ny):
        #Finds the conjugate values across the z = 0 plane
        f[j,0] = conj(f[ny-j,0])



turb_3d( 0, 1.13, 0, 0.565, 0, 0.565, 32, 16, 16)
