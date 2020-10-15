using Pkg
Pkg.activate(".")

src_dir = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(src_dir, "ctmrg_isotropic.jl"))
include(joinpath(src_dir, "2d_classical_ising.jl"))

# Make Ising model MPO
β = 1.1 * βc
d = 2
s = Index(d, "Site")
sₕ = addtags(s, "horiz")
sᵥ = addtags(s, "vert")

T = ising_mpo(sₕ, sᵥ, β)

χ0 = 1
l = Index(χ0, "Link")
lₕ = addtags(l, "horiz")
lᵥ = addtags(l, "vert")

# Initial CTM
Cₗᵤ = ITensor(lᵥ, lₕ)
Cₗᵤ[1, 1] = 1.0

# Initial HRTM
Aₗ = ITensor(lᵥ, lᵥ', sₕ)
Aₗ[lᵥ => 1, lᵥ' => 1, sₕ => 1] = 1.0

Cₗᵤ, Aₗ = ctmrg(T, Cₗᵤ, Aₗ; χmax = 20, nsteps = 100)

lᵥ = commonind(Cₗᵤ, Aₗ)
lₕ = uniqueind(Cₗᵤ, Aₗ)

Aᵤ = replaceinds(Aₗ, lᵥ => lₕ, lᵥ' => lₕ', sₕ => sᵥ)

ACₗ = Aₗ * Cₗᵤ * dag(Cₗᵤ')

ACTₗ = prime(ACₗ * dag(Aᵤ') * T * Aᵤ, -1)

κ = (ACTₗ * dag(ACₗ))[]

@show κ, exp(-β * ising_free_energy(β))

# Calculate magnetization
Tsz = ising_mpo(sₕ, sᵥ, β; sz = true)
ACTszₗ = prime(ACₗ * dag(Aᵤ') * Tsz * Aᵤ, -1)
m = (ACTszₗ * dag(ACₗ))[] / κ
@show m, ising_magnetization(β)

