using Pkg
Pkg.activate(".")

include(joinpath(@__DIR__, "..", "src", "trg.jl"))
include(joinpath(@__DIR__, "..", "src", "2d_classical_ising.jl"))

# Make Ising model MPO
β = 1.1 * βc
d = 2
s = Index(d)
l = addtags(s, "left")
r = addtags(s, "right")
u = addtags(s, "up")
d = addtags(s, "down")
T = ising_mpo((l, r), (u, d), β)

χmax = 20
nsteps = 20
κ, T, (l, r), (u, d) = trg(T, (l, r), (u, d);
                           χmax = χmax, nsteps = nsteps,
                           svd_alg = "divide_and_conquer")

κ_exact = exp(-β * ising_free_energy(β))
@show κ, κ_exact
@show abs(κ - κ_exact)

