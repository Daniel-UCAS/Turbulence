import numpy as np
from numpy import sin, cos, sqrt, ones, zeros, pi, arange

#------------------------------------------------------------------------------

def passot_pouquet_spectrum(k):
    up = 0.5
    l11 = 0.3
    ke = np.sqrt(2*np.pi)/l11    
    #print('Ke = %.4f' % ke)
    C = np.sqrt(2/np.pi)*np.power(up,2.0)/ke*32.0/3.0
    E = C*np.power(k/ke,4.0)*np.exp(-2.0*np.power(k/ke,2.0))
    return E

#------------------------------------------------------------------------------
def rndini2d(xmin, xmax,ymin,ymax,nx,ny,nz,nmodes,wn1,especf):
  nw = min(nx,ny)
  nh = nw//2 + 1
  i_cmplx = [0.0, 1.0]
  eps = np.sqrt(2*np.pi)/0.3#from spectrum/10000

  eps_new = 2*np.pi/min((ymax-ymin),(xmax-xmin))

  kcx =np.real( nh - 1 ) * 2.0 * pi //max((xmax - xmin),(ymax-ymin))
  kc = kcx*(10**-8)
  amp = 2.0*np.pi/sqrt((xmax-xmin)*(ymax-ymin))
  dx = float(xmax/nx)
  dy = float(ymax/ny)  
  dz = float(ymax/nz)
  # highest wave number that can be represented on this grid (nyquist limit)
  wnn = max(np.pi/dx, max(np.pi/dy, np.pi/dz))
  kc = wnn
  km = wn1
  #print( 'I will generate data up to wave number: ', wnn)
  
  # wavenumber step
  dk = (wnn-wn1)/nmodes
  
  # wavenumber at cell centers
  wn = wn1 + 0.5*dk + arange(0,nmodes)*dk

  u_ = zeros([nx,ny])
  v_ = zeros([nx,ny])
  i = 1
  j = 1
  for i in range(i, nh):
    for j in range(j, nw):
      ps =  2.0*pi*np.random.uniform(0.0,1.0,nmodes)
     #print (ps)
      #find local wavenumber
      kx = np.real( i - 1 ) * 2.0 * np.pi / ( xmax - xmin )
      ky = np.real( j - 1 ) * 2.0 * np.pi / ( ymax - ymin )
     # print (kx)
      if (j > nh):
         ky = -1*np.real(nw+1-j)*2. * np.pi / ( ymax - ymin ) 
      #print (ky)
            
      # get the modes   
      km1 =wnn

      espec = especf(km)
      espec = espec.clip(0.0)
      E_ = espec

      # generate turbulence at cell centers
     # um = sqrt(espec*dkn)

      #w_ = zeros([nx,ny,nz])  
      #print("i got to here") 

      #normal case
      print ("eps", eps_new)
      print("km", km)
      print("kc", kc)
      if ((eps_new<km) & (km<kc)):
        if((eps<abs(kx)) & (eps<abs(ky))):
          print("got here")
          u_[i,j] = amp*sqrt(espec*np.power(ky,2)/(np.pi*pow(km,3)) * np.exp((i_cmplx*ps))) 
          v_[i,j] = -1*kx*u_[i,j]/ky
      
        #small kx wavenumber
        if((abs(kx)<eps) & (eps<abs(ky))):
          u_[i,j] = amp*sqrt(espec*np.power(ky,2)/(np.pi*pow(km,3)) * np.exp((i_cmplx*ps))) 
        
        #small ky wavenumber
        if((abs(kx)>eps) & (abs(ky)<eps)):
          v_[i,j] = amp*sqrt(espec*np.power(kx,2)/(np.pi*pow(km,3)) * np.exp((i_cmplx*ps)))         

  j = nh+1
 # print (j)
  for j in range(j,nw):
    u_[1,j] = np.conj(u_[1,nw+2-j])
    v_[1,j] = np.conj(v_[1,nw+2-j])
  #print (v_)
  i = 1
  j = 1
  for i in range(nh):
    for j in range(nw):
      kx = np.real(i-1)*2.0*np.pi/(xmax-xmin)
      ky = np.real(j-1)*2.0*np.pi/(ymax-ymin)

      if(j>nh):
        ky = -1*np.real(nw+1-j)*2.0*np.pi/(ymax-ymin)
      
      #if((abs(np.real(kx*u_[i,j]+ky*v_[i,j]))>pow(1, -5)) & (abs(np.imag(kx*u_[i,j]+ky*v_[i,j]))>pow(1, -5))):
  print("i am done")
  return u_, v_, E_, km
U, V, E, K = rndini2d(0,5,0,10,32,32,1,1000,2*18e-6,passot_pouquet_spectrum)
#print (U)
print(2**-8)
