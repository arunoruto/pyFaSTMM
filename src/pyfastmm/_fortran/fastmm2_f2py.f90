!
!  f2py-facing wrapper for FaSTMM2, modeled on external/fastmm2/src/main.f90's
!  orchestration logic (minus command-line parsing and HDF5 file I/O -- see
!  meson.build for why io.f90 isn't part of this build at all). Geometry
!  goes in and Mueller/Jones/cross-section/T-matrix data comes out as plain
!  arrays; no files are read or written.
!
!  v1 covers spherical (Lorenz-Mie) monomers only -- not the
!  precomputed-per-monomer-T-matrix ("arbitrarily-shaped constituent
!  particles") input path upstream also supports.
!
module fastmm2_f2py_bindings
use common
use octtree
use interpolation
use matvec
use solver
use mie
use orientation_averaging
use T_matrix
implicit none

!
!  Module-level variable so the Python side can choose the truncation
!  formula before calling solve()/compute_tmatrix(). 0 = default
!  (truncation_order, floor(ka+3*ka^(1/3))), 1 = conservative
!  (truncation_order2, larger order for a given ka -- typically closer
!  to Wiscombe, especially for small ka where near-field coupling of
!  touching spheres needs more terms).
!
integer, save :: truncation_formula = 0

contains

!
!  Build the sphere(:) array (data_struct, internal-only -- never an f2py
!  dummy argument) from plain coordinate/radius/permittivity arrays, and
!  center it the same way main.f90 does before octree construction.
!  Shared by fastmm2_solve_fixed / fastmm2_solve_averaged /
!  fastmm2_compute_tmatrix.
!
subroutine build_spheres(n, coords, radii, eps_r, eps_i, k, sphere, max_sph)
    implicit none
    integer, intent(in) :: n
    real(dp), intent(in) :: coords(3,n), radii(n), eps_r(n), eps_i(n)
    complex(dp), intent(in) :: k
    type(data_struct), allocatable, intent(out) :: sphere(:)
    integer, intent(out) :: max_sph

    real(dp) :: coord(3,n)
    integer :: sph, max_loc(3), min_loc(3)
    real(dp) :: cc(3), ka

    coord = coords

    max_loc(1) = maxloc(coord(1,:),1)
    max_loc(2) = maxloc(coord(2,:),1)
    max_loc(3) = maxloc(coord(3,:),1)

    min_loc(1) = minloc(coord(1,:),1)
    min_loc(2) = minloc(coord(2,:),1)
    min_loc(3) = minloc(coord(3,:),1)

    cc(1) = ((coord(1,max_loc(1)) + radii(max_loc(1))) + &
         (coord(1,min_loc(1)) - radii(min_loc(1))))/2.0d0
    cc(2) = ((coord(2,max_loc(2)) + radii(max_loc(2))) + &
         (coord(2,min_loc(2)) - radii(min_loc(2))))/2.0d0
    cc(3) = ((coord(3,max_loc(3)) + radii(max_loc(3))) + &
         (coord(3,min_loc(3)) - radii(min_loc(3))))/2.0d0

    do sph = 1, n
       coord(:,sph) = coord(:,sph) - cc
    end do

    allocate(sphere(n))
    max_sph = 0

    do sph = 1, n
       sphere(sph)%cp = coord(:,sph)
       sphere(sph)%r = radii(sph)
       sphere(sph)%eps_r = dcmplx(eps_r(sph), eps_i(sph))

       ka = real(k) * radii(sph)
       if (truncation_formula == 1) then
          sphere(sph)%Nmax = truncation_order2(ka)
       else
          sphere(sph)%Nmax = truncation_order(ka)
       end if

       sphere(sph)%Tmat_ind = 0
       sphere(sph)%ifT = 0
       sphere(sph)%euler_angles = 0.0_dp

       if (max_sph < sphere(sph)%Nmax) max_sph = sphere(sph)%Nmax
    end do
end subroutine build_spheres

!
!  Build the octree and (for form /= 0, i.e. FaSTMM/FaSTMM2) the MLFMM
!  interpolation/translation operators, exactly as main.f90 does before
!  either solving or computing a T-matrix. Returns the cluster-level
!  truncation order Nmax used by rhs2_xy/gmres_mlfmm2/inc_xy/mueller_matrix.
!
subroutine build_octree(sphere, k, formulation, acc, otree, nmax_cluster)
    implicit none
    type(data_struct), dimension(:) :: sphere
    complex(dp), intent(in) :: k
    integer, intent(in) :: formulation, acc
    type(level_struct), dimension(:), allocatable, intent(out) :: otree
    integer, intent(out) :: nmax_cluster

    integer :: max_level
    real(dp) :: ka

    call create_octtree(sphere, otree, dble(k), max_level, formulation, acc)

    if (formulation /= 0) then
       call build_interpolation_matrix_mlfma2(otree)
       call compute_translators_mlfma2(otree, k)
       call initialize_samples(otree, k)
    end if

    ka = dble(k)*sqrt(3.0d0)/2.0d0 * otree(1)%tree(1)%dl
    if (truncation_formula == 1) then
       nmax_cluster = truncation_order2(ka)
    else
       nmax_cluster = truncation_order(ka)
    end if
end subroutine build_octree

!
!  Fixed-orientation solve: builds spheres + octree, runs the GMRES/MLFMM
!  solve, and returns the Mueller matrix, Jones matrix, and cross sections
!  -- mirrors main.f90's N_ave == 0 branch.
!
!    n            -- number of spheres
!    coords(3,n)  -- sphere center coordinates
!    radii(n)     -- sphere radii
!    eps_r(n), eps_i(n) -- real/imaginary parts of sphere electric permittivity
!    k_in         -- wavenumber (real; medium is assumed non-absorbing, as
!                    upstream's own CLI always passes a purely real k)
!    N_theta, N_phi -- angular resolution of the returned Mueller/Jones matrices
!    formulation  -- 0: STMM, 1: FaSTMM, 2: FaSTMM2
!    acc          -- desired FMM accuracy (number of significant digits)
!    tol, restart, max_iter -- GMRES parameters
!
!  Returns:
!    mueller(N_theta*N_phi, 18) -- columns: [phi, theta, P11..P44]
!    jones(N_theta*N_phi, 6)    -- columns: [phi, theta, S1, S2, S3, S4]
!                                  (complex; phi/theta carried as
!                                  zero-imaginary-part complex values, same
!                                  convention as upstream's own jones.h5)
!    cross_sections(5)          -- [Cext, Cext-Cabs, Cabs, Csca, asymmetry parameter]
!
subroutine fastmm2_solve_fixed(n, coords, radii, eps_r, eps_i, k_in, &
        N_theta, N_phi, formulation, acc, tol, restart, max_iter, &
        mueller, jones, cross_sections)
    implicit none
    integer, intent(in) :: n, N_theta, N_phi, formulation, acc, restart, max_iter
    real(dp), intent(in) :: coords(3,n), radii(n), eps_r(n), eps_i(n)
    real(dp), intent(in) :: k_in, tol
    real(dp), intent(out) :: mueller(N_phi*N_theta, 18)
    complex(dp), intent(out) :: jones(N_phi*N_theta, 6)
    real(dp), intent(out) :: cross_sections(5)

    type(data_struct), allocatable :: sphere(:)
    type(level_struct), allocatable :: otree(:)
    type(Tmatrix), allocatable :: Tmat(:)
    complex(dp) :: k
    complex(dp), allocatable :: b_vec(:), b_vec2(:), xx(:), xx2(:)
    real(dp), allocatable :: S_out(:,:)
    complex(dp), allocatable :: J_out(:,:)
    integer :: nmax_cluster, max_sph

    k = dcmplx(k_in, 0.0d0)
    allocate(Tmat(0))

    call build_spheres(n, coords, radii, eps_r, eps_i, k, sphere, max_sph)
    call build_octree(sphere, k, formulation, acc, otree, nmax_cluster)

    call rhs2_xy(nmax_cluster, sphere, Tmat, b_vec, b_vec2, k)

    allocate(xx(size(b_vec)))
    allocate(xx2(size(b_vec)))

    call gmres_mlfmm2(sphere, otree, Tmat, k, b_vec, b_vec2, xx, xx2, tol, restart, max_iter)

    deallocate(b_vec, b_vec2)
    call inc_xy(nmax_cluster, sphere, b_vec, b_vec2, k)

    call mueller_matrix(sphere, xx, xx2, b_vec, b_vec2, k, N_theta, N_phi, &
        nmax_cluster, S_out, J_out, cross_sections, 0.0_dp, 0.0_dp)

    mueller = S_out
    jones = J_out
end subroutine fastmm2_solve_fixed

!
!  Orientation-averaged solve: same inputs as fastmm2_solve_fixed, plus
!  N_ave (number of orientations) and halton_init (starting point of the
!  Halton sequence used to generate them) -- mirrors main.f90's N_ave > 0
!  branch. No Jones matrix is produced in this mode (matches upstream).
!
!  Returns:
!    mueller(N_theta, 17) -- columns: [scattering angle, P11..P44]
!    cross_sections(5)    -- [Cext, Cext-Cabs, Cabs, Csca, asymmetry parameter]
!
subroutine fastmm2_solve_averaged(n, coords, radii, eps_r, eps_i, k_in, &
        N_theta, N_phi, N_ave, halton_init, formulation, acc, tol, restart, max_iter, &
        mueller, cross_sections)
    implicit none
    integer, intent(in) :: n, N_theta, N_phi, N_ave, halton_init
    integer, intent(in) :: formulation, acc, restart, max_iter
    real(dp), intent(in) :: coords(3,n), radii(n), eps_r(n), eps_i(n)
    real(dp), intent(in) :: k_in, tol
    real(dp), intent(out) :: mueller(N_theta, 17)
    real(dp), intent(out) :: cross_sections(5)

    type(data_struct), allocatable :: sphere(:)
    type(Tmatrix), allocatable :: Tmat(:)
    complex(dp) :: k
    real(dp), allocatable :: S_ave(:,:)
    integer :: max_sph

    k = dcmplx(k_in, 0.0d0)
    allocate(Tmat(0))

    call build_spheres(n, coords, radii, eps_r, eps_i, k, sphere, max_sph)

    call orientation_ave(sphere, Tmat, 0, k, N_phi, N_theta, N_ave, halton_init, &
        S_ave, cross_sections, formulation, tol, restart, max_iter, acc)

    mueller = S_ave
end subroutine fastmm2_solve_averaged

!
!  T-matrix computation: builds spheres + octree directly into
!  compute_T_matrix, skipping the fixed-orientation solve+Mueller step
!  main.f90's CLI flow does as a side effect (compute_T_matrix only needs
!  sphere/otree/Tmat already built, confirmed by reading its signature/body
!  -- it does not consume a prior solve's output).
!
!    T_order -- requested T-matrix truncation order
!    nm      -- (T_order+1)**2-1, the T-matrix dimension, computed by the
!               caller and passed in explicitly. f2py's C-code generator
!               mishandles a "**" exponent inside an intent(out) array's
!               dimension expression (confirmed: "(T_order+1)**2-1" used
!               directly here produces invalid generated C, "* *2" instead
!               of computing the square) -- dimensioning off a plain extra
!               argument instead, exactly as we're forced to elsewhere for
!               array sizes that can't be inferred from module state, works
!               around it.
!  (all other inputs match fastmm2_solve_fixed/averaged)
!
!  Returns Taa, Tab, Tba, Tbb, each nm x nm, ordered as
!  (-1(1),0(1),1(1),-2(2),...,N(N)) per monomer/cluster convention (same
!  layout as upstream's own T-matrix HDF5 output).
!
subroutine fastmm2_compute_tmatrix(n, coords, radii, eps_r, eps_i, k_in, &
        T_order, nm, formulation, acc, tol, restart, max_iter, &
        Taa, Tab, Tba, Tbb)
    implicit none
    integer, intent(in) :: n, T_order, nm, formulation, acc, restart, max_iter
    real(dp), intent(in) :: coords(3,n), radii(n), eps_r(n), eps_i(n)
    real(dp), intent(in) :: k_in, tol
    complex(dp), intent(out) :: Taa(nm, nm)
    complex(dp), intent(out) :: Tab(nm, nm)
    complex(dp), intent(out) :: Tba(nm, nm)
    complex(dp), intent(out) :: Tbb(nm, nm)

    type(data_struct), allocatable :: sphere(:)
    type(level_struct), allocatable :: otree(:)
    type(Tmatrix), allocatable :: Tmat(:)
    complex(dp) :: k
    integer :: nmax_cluster, max_sph

    k = dcmplx(k_in, 0.0d0)
    allocate(Tmat(0))

    call build_spheres(n, coords, radii, eps_r, eps_i, k, sphere, max_sph)
    call build_octree(sphere, k, formulation, acc, otree, nmax_cluster)

    call compute_T_matrix(sphere, otree, Tmat, k, T_order, tol, restart, max_iter, &
        Taa, Tab, Tba, Tbb)
end subroutine fastmm2_compute_tmatrix

end module fastmm2_f2py_bindings
