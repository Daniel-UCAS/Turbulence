import numpy as np
from numpy import sin, cos, sqrt, ones, zeros, pi, arange, conj, convolve
import matplotlib.pyplot as plt
from numpy.fft import fftn
#------------------------------------------------------------------------------
def passot_pouquet_spectrum(k,l):
    up = 0.5
    l11 = l/3
    ke = np.sqrt(2*np.pi)/l11    
    #print('Ke = %.4f' % ke)
    C = np.sqrt(2/np.pi)*np.power(up,2.0)/ke*32.0/3.0
    E = C*np.power(k/ke,4.0)*np.exp(-2.0*np.power(k/ke,2.0))
    return E
#------------------------------------------------------------------------------

def turb_3d(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz):
    i_cmplx = 1j
    delta = np.max(((xmax-xmin)/(nx-1)) and ((ymax-ymin)/(ny-1)) and ((zmax-zmin)/(nz-1)))
    kc = np.pi/delta
    amp = sqrt((2.0*np.pi)**3/((xmax-xmin) * (ymax-ymin) * (zmax-zmin)))
    uk = zeros([nx,ny,nz,3],dtype=np.complex_)
    for k in range(0,nz):
        if ((2*int(nz/2) == nz) and (k == 1+nz/2)):
            continue        
        for j in range(0,ny):  
            if ((2*int(ny/2) == ny) and (j == 1+ny/2)):
                continue
            for i in range(0,nx):
                if ((i == 0) and (j == 0) and (k == 0)):
                    continue
                if ((2*int(nx/2) == nx) and (i == 1+nx/2)):
                    continue
                ps1   = np.random.uniform(-np.pi,np.pi)
                ps2   = np.random.uniform(-np.pi,np.pi)
                psr   = np.random.uniform(0,2.0*np.pi)
                kx = np.real(i-1)*2.0*np.pi/(xmax-xmin)
                if (j <= 1 + ny/2):
                    ky = np.real(j-1)*2.0*np.pi/(ymax-ymin)
                else:
                    ky = -1.0*np.real(ny+1-j)*2.0*np.pi/(ymax-ymin)
                if (k <= 1 + nz/2):
                    kz = np.real(k-1)*2.0*np.pi/(zmax-zmin)
                else:
                    kz = -1.0*np.real(nz+1-k)*2.0*np.pi/(zmax-zmin)
                
                k2 = sqrt(kx**2 + ky**2)
                kmag = sqrt(kx**2 + ky**2 + kz**2)
                
                if (kmag > kc):
                    continue
                e_spec = passot_pouquet_spectrum(kmag,xmax)
                ak = amp*sqrt(e_spec/(2.0*np.pi*kmag**2))*np.exp(i_cmplx*ps1)*cos(psr)
                bk = amp*sqrt(e_spec/(2.0*np.pi*kmag**2))*np.exp(i_cmplx*ps2)*sin(psr)

                if(k2 < (np.sqrt(2*np.pi)/(xmax/3))/1e4):
                    uk[i, j, k, 0] = (ak+bk)/sqrt(2.0)
                else:
                    uk[i, j, k, 0] = (ak*kmag*ky+bk*kx*kz)/(kmag*k2)
                if(k2 < (np.sqrt(2*np.pi)/(xmax/3))/1e4): 
                    uk[i, j, k, 1] = (bk-ak)/sqrt(2.0)
                else:
                    uk[i, j, k, 1] = (bk*ky*kz-ak*kmag*kx)/(kmag*k2)
                uk[i, j, k, 2] = -(bk*k2)/kmag
 

    
    conjugate_yzplane(uk[0,:,:,0],nx,ny,nz,0)
    conjugate_yzplane(uk[0,:,:,1],nx,ny,nz,1)
    conjugate_yzplane(uk[0,:,:,2],nx,ny,nz,2)
    print(uk[0,0,19,2])
    #quit()
    print(uk)
    #quit()
    z = np.fft.fft(uk)
    #print(z)
    y = np.ascontiguousarray(z.real, dtype=np.float32)
    #np.sin(2*np.pi*x)
    with open('Z.bin', 'wb') as f:
        f.write(y)


def conjugate_yzplane(f,nx,ny,nz,num):
    #Check if f is the same as f
    """for k in range(0,zpes-1):
        for j in range(0,ypes-1):
            f(j*ny+1:(j+1)*ny,k*nz+1:(k+1)*nz) = f3(1:ny,1:nz,k*ypes+j+1)
    """
    for k in range(int(2+nz/2),nz):
        f[0,k] = conj(f[1,nz+2-k-1])
        for j in range(2,ny):
            f[j,k] = conj(f[ny+2-j-1,nz+2-k-1])
    for j in range(int(2+ny/2),ny):
        f[j,0] = conj(f[ny+2-j-1,1])
    """for k in range(0,zpes-1):
        for j in range(0,ypes-1):
            f3(1:ny,1:nz,k*ypes+j+1) = f(j*ny+1:(j+1)*ny,k*nz+1:(k+1)*nz)
    """
    print(f[0,19])

turb_3d( 0, 0.565, 0, 0.565, 0, 0.565, 32, 32, 32)