using Pkg
Pkg.activate(".")

using ITensors
include(joinpath("..", "..", "src", "2d_classical_ising.jl"))
include(joinpath("..", "..", "src", "ctmrg_anisotropic.jl"))

# Unit cell size
ny, nx = 2, 2

# Make the site indices
sh, sv = site_inds(ny, nx, 2)

# Make the Ising partition function
β = 1.1 * βc
T = ising_partition(sh, sv, β)

# Make the initial environment
(Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad) = ctmrg_environment((sh, sv))

# Check the initialize environment
check_environment(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))

# Run ctmrg
@show κave = ctmrg(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))

@assert isapprox(κave, exp(-β * ising_free_energy(β)); rtol=1e-10)
