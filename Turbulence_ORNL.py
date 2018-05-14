!=========================================================================================
  subroutine rndini2d(xmin,xmax,ymin,ymax,nx,ny,nz,x,seed,u,io)
!=========================================================================================
! compute the turbulence flow field based on the VKP energy spectrum.
!
! NOTE: This routine assumes that the 2-D plane is the XY plane.
! It will NOT WORK for YZ or XZ configurations (though this could be
! fixed without too much work - just some more logic)
!
!-----------------------------------------------------------------------------------------
!  use param_m, only : ny_g  ! really this should all be cleaned up...
  use topology_m

  implicit none
!-----------------------------------------------------------------------------------------
! declarations passed in

  real, intent(in) :: xmax, xmin, ymin, ymax    !grid start and end locations
  integer, intent(in) :: nx,ny,nz               !grid dimensions
  real, intent(in) :: x(nx)                     !grid locations
  real u(nx,ny,nz,3)                            !velocity field

  integer seed, io                              !turbulence seed, io unit

! local declarations

  real    amp
  real    c
  real    e_spec, eps
  real    intl1, intl2
  real    kc, kx, ky
  real    mag_k
  real    pao
  real    pi
  real    ps
  real    r11, r22
  real    rat_k, rat2_k
  real    u1m,u2m
  real    urms
  real    vk

  complex i_cmplx
  complex u1k(nh,nw)                              !fourier space x-velocity
  complex u2k(nh,nw)                              !fourier space y-velocity

  integer i, j, k, l
  integer ii,   jj
  integer ivar, npoints

  real eps_new ! minumum filled wavenumber (Evatt 11-JUN-2004)
!-----------------------------------------------------------------------------------------
! Revisions and comments:
! 1. The routine computes the turbulence flow field
! based on the VKP energy spectrum:
! E(k) = A_k (Up^5/dissip) * (k/k_e)^4/(1+(k/k_e)^2)^(17/6) exp(-(3/2) a_k (k/k_d)^(4/3) )
! where A_k = 1.5
! a_k = 1.5
! Up  = rms velocity
! k_e = (in code Ke ) is the wave number associated
!  with most energetic scales
! k_d = (in code Kd ) is the wave number associated
!  with dissipation scales
!
! 2. The spectrum is specified inside a range Eps < k < Kc,
!   where Eps = is a very small number set to near zero (Ke / 1000, here).
! Kc  = cut-off scale.
!   Outside this range the spectrum E(k) is set to zero.
!
! 3. The descrete wave numbers are based on all possible descrete periods along
!   a given direction ranging from 0 to (N-1)*2*pi/L.
!
! 4. Since the range L is not specified exactly to yield
!   a multiple of 2*pi, a truncation of the ratio L/(2*pi)
!   is carried out to avoid out-of-range computations.
!-----------------------------------------------------------------------------------------
! set pi

  pi = 4.0 * atan ( 1.0 )
!-----------------------------------------------------------------------------------------
! error checking

!  Evatt Hawkes: this appears to be a legacy from when nw was read from an input file
!  so have cut it
!  if( nw .gt. ny_g/2 ) then
!    if( myid.eq.0 ) then
!      write(io,9350)
!    endif
!    call terminate_run(io,0)  !must be called by all processors
!  endif

! Evatt Hawkes JUN 2004 : previously ke and kd were entered as lengths
! and juggled back and forth between wavenumber and length - that was confusing
! transform ke and kd into a wavenumber (note: angular wavenumber)
!  ke = 2.0 * pi / ke
!  kd = 2.0 * pi / kd

! initialize field with homogeneous turbulence on top of laminar solution

  i_cmplx = ( 0.0, 1.0 )

! specify the cut-offs for the evaluation of spectrum
! eps means a `small number'

  eps = ke / 10000.0

! MODIFICATION OF EPS (Evatt 11-JUN-2004)
! want to allow no bigger waves than the smallest dimension
  eps_new = 2.0*pi/min((ymax-ymin),(xmax-xmin))

! Xmax-Xmin > -2 Ymin  ???note from evatt - did not understand this comment

! cut-off wavenumber - must be less than the Nyquist cut-off
! evatt has set this equal to the Nyquist cut-off by setting nh=nx_g/2+1
! i am not sure how general this is for varying box sizes
  kc = real ( nh - 1 ) * 2.0 * pi /max((xmax - xmin),(ymax-ymin))

! truncation to 8 digits only ???note from evatt - did not understand this comment
! Kc = 1.E-8*int(Kc * 1.e08)

! set constants for vkp spectrum
! Evatt Hawkes JUN 2004 - c no longer needed now there is a general spectrum call
!  c  = 1.5 * up**5 / dissip
  amp = 2.0 * pi / sqrt ( (xmax-xmin)*(ymax-ymin) )

! loop over fourier space
! note that all processes initialise the same 2D domain in fourier space
! i.e. there is no domain decomposition in fourier space!
  do i = 1, nh
    do j = 1, nw

!     set a random phase
      ps = 2.0 * pi * ( ran2 ( seed ) - 0.5 )

!     find local wavenumber
      kx = real ( i - 1 ) * 2. * pi / ( xmax - xmin )
      ky = real ( j - 1 ) * 2. * pi / ( ymax - ymin )

!     by convention negative wavenumbers are stored in reverse starting from nh+1
      if( j .gt. nh ) then
        ky = - real(nw+1-j)*2. * pi / ( ymax - ymin )
      endif

!     wavenumber magnitude
      mag_k = sqrt ( kx**2 + ky**2 )


!     Evatt Hawkes JUN 2004 - no longer needed now there is a general spectrum
!!     ratio of wavenumber to "energetic" wavenumber
!      rat_k = mag_k / ke
!
!!     ratio of wavenumber to "dissipation" wavenumber
!      rat2_k = mag_k / kd
!
!!     spectrum
!      vk = c * rat_k**4 / ( 1.0 + rat_k**2 )**(17./6.)
!      pao = exp ( -2.25 * rat2_k**(4.0/3.0) )
!      e_spec = vk * pao

      e_spec = energy_spectrum(mag_k)

!     intialise velocity
      u1k(i,j) = 0.0
      u2k(i,j) = 0.0

!     only set if within a sensible range
!     zero wavenumber not filled as this would result in a net velocity
!     wavenumbers greater than the cut-off wavenumber not filled

!        Evatt temporary mod 11-JUN-2004
!      if( eps .lt. mag_k .and. mag_k .lt. kc ) then
      if( eps_new .lt. mag_k .and. mag_k .lt. kc ) then

!       normal case
        if( eps .lt. abs(kx) .and. eps .lt. abs(ky)) then
!         set velocity from known magnitude from spectrum, and random phase
          u1k(i,j) = amp * sqrt ( e_spec * ky**2 / ( pi * mag_k**3 ) )  &
                   * exp ( i_cmplx * ps )
!         enforce continuity in fourier space (k.u=0)
          u2k(i,j) = - kx * u1k(i,j)/ky
        endif

!       dealing with small kx wavenumber
        if( abs(kx) .lt. eps .and. abs(ky) .gt. eps ) then
!         set velocity from known magnitude from spectrum, and random phase
          u1k(i,j) = amp * sqrt ( e_spec * ky**2 / ( pi * mag_k**3 ) )  &
                   * exp ( i_cmplx * ps )
!         since u2k=0 and kx=0 continuity is satisfied
        endif

!       dealing with small ky wavenumber
        if( abs(kx) .gt. eps .and. abs(ky) .lt. eps ) then
!         set velocity from known magnitude from spectrum, and random phase
          u2k(i,j) = amp * sqrt ( e_spec * kx**2 / ( pi * mag_k**3 ) )  &
                   * exp ( i_cmplx * ps )
!         since u1k=0 and ky=0 continuity is satisfied
        endif

      endif

    enddo
  enddo

! since transformed result is real, fourier space result must be conjugate symmetric
! enforce j-symmetry here for i=1.  i-symmetry is later enforced for all j in rin2dy
  do j = nh+1, nw
    u1k(1,j) = conjg ( u1k(1,nw+2-j) )
    u2k(1,j) = conjg ( u2k(1,nw+2-j) )
  enddo

! do a check that we actually achieved zero divergence in fourier space,
! i.e. that continuity will be satisfied in physical space
  do i = 1, nh
    do j = 1, nw

      kx = real ( i - 1 ) * 2.0 * pi / ( xmax - xmin )
      ky = real ( j - 1 ) * 2.0 * pi / ( ymax - ymin )

      if( j .gt. nh ) then
        ky =-real(nw+1-j) * 2.0 * pi / ( ymax - ymin )
      endif

      if(abs(real(kx*u1k(i,j)+ky*u2k(i,j))) .gt. 1.e-5 .or.         &
      abs(aimag(kx*u1k(i,j)+ky*u2k(i,j))) .gt. 1.e-5 ) then
        if(myid.eq.0) then
          write(io,4010) i,j, myid
        endif
      endif

    enddo
  enddo

! do the inverse transformations etc

  ivar = 1
  call rin2dy ( ivar, nx,ny,u1k, xmin, xmax, x, u(:,:,1,ivar) )

  ivar = 2
  call rin2dy ( ivar, nx,ny,u2k, xmin, xmax, x, u(:,:,1,ivar) )
!-----------------------------------------------------------------------------------------
! format statements

  4010 format(' non zero divergence in fourier space :',3i4)
!  Evatt Hawkes removed July 2003
!  9350 format(/' error: wrong specification of nw')
!-----------------------------------------------------------------------------------------
  return
  end subroutine rndini2d