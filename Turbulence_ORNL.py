import numpy as np
import random as ran
from numpy import sin, cos, sqrt, ones, zeros, pi, arange

#------------------------------------------------------------------------------

def passot_pouquet_spectrum(k):
    up = 0.5
    l11 = 0.6*(1**-3)
    ke = np.sqrt(2*np.pi)/l11    
    #print('Ke = %.4f' % ke)
    C = np.sqrt(2/np.pi)*np.power(up,2.0)/ke*32.0/3.0
    E = C*np.power(k/ke,4.0)*np.exp(-2.0*np.power(k/ke,2.0))
    return E

#------------------------------------------------------------------------------
def rndini2d(xmin, xmax,ymin,ymax,nx,ny,nz,nmodes,wn1,especf):
  
  #xmin = 0
  #xmax = lx
  #ymin = 0
  #ymax = ly  
  nw = min(nx,ny)//2-1
  nh = nw//2 + 1
  i_cmplx =complex(0,1)
  eps =wn1# (np.sqrt(2*np.pi)/(0.6*(10**-3)))/10000

  dx = float(xmax/nx)
  dy = float(ymax/ny) 

  #eps_new = 2*np.pi/min((ymax-ymin),(xmax-xmin))
  kc = max(np.pi/dx, np.pi/dy)
  print (kc)

  #kc = np.real( nh - 1 ) * 2.0 * pi /max((xmax - xmin),(ymax-ymin))
  amp = 2.0*np.pi/sqrt((xmax-xmin)*(ymax-ymin))
 
  
  # highest wave number that can be represented on this grid (nyquist limit)
  #wnn = max(np.pi/dx, max(np.pi/dy, np.pi/dz))
  #kc = wnn
  #km = wn1
  #print( 'I will generate data up to wave number: ', wnn)
  
  # wavenumber step
  dk = (kc-wn1)/nmodes
  
  # wavenumber at cell centers
  wn = wn1 + 0.5*dk + arange(0,nmodes)*dk

  u_ = zeros([nx,ny],dtype=np.complex_)
  v_ = zeros([nx,ny],dtype=np.complex_)
  a = 1
  b = 1
  for a in range(a, nh):
    for b in range(b, nw):
      print("num")
      ps =  2.0*np.pi*(np.random.randint(100000)-0.5)
     
      #find local wavenumber
      kx = np.real( a - 1 ) * 2.0 * np.pi / ( xmax - xmin )
      ky = np.real( b - 1 ) * 2.0 * np.pi / ( ymax - ymin )
    
      #print (kx)
      if (b > nh):
         ky = -1*np.real(nw+1-b)*2. * np.pi / ( ymax - ymin ) 
      #print (ky)
            
      # get the modes   
      #km1 =wnn
      km = ( np.sqrt((kx**2) + (ky**2)))
      espec = especf(km)
      espec = espec.clip(0.0)
      E_ = espec
      u_[a,b] = 0.0
      v_[a,b] = 0.0
      #generate turbulence at cell centers

      #normal case
      #print ("eps", eps)
      #print("km", km)
      #print("kc", kc)
      if ((eps<km) & (km<kc)):
        print("yes)")
        if((eps<abs(kx)) & (eps<abs(ky))):
          
          u_[a,b] = amp*sqrt(espec*pow(ky,2)/(np.pi*pow(km,3))) * (np.exp(i_cmplx*ps))
          v_[a,b] = -1*kx*u_[a,b]/ky
      
        #small kx wavenumber
        if((eps>abs(kx)) & (eps<abs(ky))):
          u_[a,b] = amp*sqrt(espec*pow(ky,2)/(np.pi*pow(km,3))) * (np.exp(i_cmplx*ps))
         
        
        #small ky wavenumber
        if((eps<abs(kx)) & (eps>abs(ky))):
          v_[a,b] = amp*sqrt(espec*pow(kx,2)/(np.pi*pow(km,3))) * (np.exp(i_cmplx*ps))
         
  b = nh+1
 # print(u_)
 # print (j)
  for b in range(b,nw):
    u_[1,b] = np.conj(u_[1,nw+2-b])
    v_[1,b] = np.conj(v_[1,nw+2-b])
  #print (v_)
  a = 1
  b = 1
  for a in range(nh):
    for b in range(nw):
      kx = np.real(a-1)*2.0*np.pi/(xmax-xmin)
      ky = np.real(b-1)*2.0*np.pi/(ymax-ymin)

      if(b>nh):
        ky = -1*np.real(nw+1-b)*2.0*np.pi/(ymax-ymin)
      
      #if((abs(np.real(kx*u_[i,j]+ky*v_[i,j]))>pow(1, -5)) & (abs(np.imag(kx*u_[i,j]+ky*v_[i,j]))>pow(1, -5))):
  print("i am done")
  return u_, v_, E_, km
U, V, E, K = rndini2d(0,1,0,1,16,16,1,1000,2*18e-6,passot_pouquet_spectrum)

