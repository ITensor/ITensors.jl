using ITensors

N = 4
nmps = 3
cutoff = 1e-8
s = siteinds("S=1/2", N)
ψs = [randomMPS(s; linkdims=2) for _ in 1:nmps]
ρs = [outer(ψ, ψ; cutoff) for ψ in ψs]
ρ = sum(ρs; cutoff)

ψs_full = prod.(ψs)
ρs_full = [ψ'dag(ψ) for ψ in ψs_full]
ρ_full = sum(ρs_full)
@show norm(ρ_full - prod(ρ))
