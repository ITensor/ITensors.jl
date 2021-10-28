using ITensors
using ITensors.ITensorNetworkMaps
using KrylovKit
using LinearAlgebra

include("utils.jl")

N = 4
s = siteinds("S=1/2", N; conserve_qns=true)
χ = 2
ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=χ)
ℋ = OpSum()
for n in 1:(N - 1)
  ℋ .+= 0.5, "S+", n, "S-", n + 1
  ℋ .+= 0.5, "S-", n, "S+", n + 1
  ℋ .+= "Sz", n, "Sz", n + 1
end
H = MPO(ℋ, s)

T = [transfer_matrix(ψ, n) for n in 1:N]
Tᴴ = [transfer_matrix((H, ψ), n) for n in 1:N]
@show (T[1] * T[2] * T[3] * T[4] * ITensor(1))[] ≈ inner(ψ, ψ)
@show (ITensor(1) * T[1] * T[2] * T[3] * T[4])[] ≈ inner(ψ, ψ)
@show (ITensor(1) * (T[1] * T[2] * T[3] * T[4]))[] ≈ inner(ψ, ψ)
@show contract(T[1] * T[2] * T[3] * T[4])[] ≈ inner(ψ, ψ)

@show contract(prod(T))[] ≈ inner(ψ, ψ)
@show contract(prod(Tᴴ))[] ≈ inner(ψ, H, ψ)

Ttot = transfer_matrix(ψ, 1:N)
Tᴴtot = transfer_matrix((H, ψ), 1:N)
@show contract(Ttot)[] ≈ inner(ψ, ψ)

println("Time to contract transfer matrix")
@time contract(Tᴴtot)[]
println("Time to contract using `inner(ψ, H, ψ)`")
@time inner(ψ, H, ψ)
@show contract(Tᴴtot)[] ≈ inner(ψ, H, ψ)

v = cache(T)
vᴴ = cache(Tᴴ)

@show (v[2 => 3] * v[3 => 2])[] ≈ inner(ψ, ψ)
@show (v[1 => 2] * v[2 => 1])[] ≈ inner(ψ, ψ)

@show (vᴴ[2 => 3] * vᴴ[3 => 2])[] ≈ inner(ψ, H, ψ)
@show (vᴴ[1 => 2] * vᴴ[2 => 1])[] ≈ inner(ψ, H, ψ)

# One-site effective Hamiltonian
H² = hessian(H, vᴴ, 2)
ψ² = ψ[2]
@show (dag(ψ²) * H²(ψ²))[] ≈ inner(ψ, H, ψ)

# Two-site effective Hamiltonian
H²⁻³ = hessian(H, vᴴ, 2:3)
ψ²⁻³ = prod(ψ[2:3])
@show (dag(ψ²⁻³) * H²⁻³(ψ²⁻³))[] ≈ inner(ψ, H, ψ)

# Three-site effective Hamiltonian
H²⁻⁴ = hessian(H, vᴴ, 2:4)
ψ²⁻⁴ = prod(ψ[2:4])
@show (dag(ψ²⁻⁴) * H²⁻⁴(ψ²⁻⁴))[] ≈ inner(ψ, H, ψ)
