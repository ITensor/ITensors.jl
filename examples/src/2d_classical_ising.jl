using ITensors
using LinearAlgebra
using QuadGK

function ising_mpo(
  pair_sₕ::Pair{<:Index,<:Index},
  pair_sᵥ::Pair{<:Index,<:Index},
  β::Real,
  J::Real=1.0;
  sz::Bool=false,
)
  sₕ, sₕ′ = pair_sₕ
  sᵥ, sᵥ′ = pair_sᵥ
  @assert dim(sₕ) == dim(sᵥ)
  d = dim(sₕ)
  T = ITensor(sₕ, sₕ′, sᵥ, sᵥ′)
  for i in 1:d
    T[i, i, i, i] = 1.0
  end
  if sz
    T[1, 1, 1, 1] = -1.0
  end
  s̃ₕ, s̃ₕ′, s̃ᵥ, s̃ᵥ′ = sim.((sₕ, sₕ′, sᵥ, sᵥ′))
  T̃ = T * δ(sₕ, s̃ₕ) * δ(sₕ′, s̃ₕ′) * δ(sᵥ, s̃ᵥ) * δ(sᵥ′, s̃ᵥ′)

  # Alternative method
  #Q = [exp(β * J) exp(-β * J); exp(-β * J) exp(β * J)]
  #X = √Q

  f(λ₊, λ₋) = [
    (λ₊ + λ₋)/2 (λ₊ - λ₋)/2
    (λ₊ - λ₋)/2 (λ₊ + λ₋)/2
  ]
  λ₊ = √(exp(β * J) + exp(-β * J))
  λ₋ = √(exp(β * J) - exp(-β * J))
  X = f(λ₊, λ₋)
  Xₕ = itensor(vec(X), s̃ₕ, sₕ)
  Xₕ′ = itensor(vec(X), s̃ₕ′, sₕ′)
  Xᵥ = itensor(vec(X), s̃ᵥ, sᵥ)
  Xᵥ′ = itensor(vec(X), s̃ᵥ′, sᵥ′)
  return T̃ * Xₕ′ * Xᵥ′ * Xₕ * Xᵥ
end

function ising_mpo(sₕ::Index, sᵥ::Index, args...; kwargs...)
  return ising_mpo(sₕ => sₕ', sᵥ => sᵥ', args...; kwargs...)
end

function ising_mpo_dual(
  sh::Tuple{Index,Index}, sv::Tuple{Index,Index}, β::Real, J::Real=1.0
)
  d = dim(sh[1])
  T = ITensor(sh[1], sh[2], sv[1], sv[2])
  sig(s) = 1.0 - 2.0 * (s - 1)
  E0 = -4.0
  for s1 in 1:d, s2 in 1:d, s3 in 1:d, s4 in 1:d
    E = sig(s1) * sig(s2) + sig(s2) * sig(s3) + sig(s3) * sig(s4) + sig(s4) * sig(s1)
    val = exp(-β * (E - E0))
    T[sh[1] => s1, sv[2] => s2, sh[2] => s3, sv[1] => s4] = val
  end
  return T
end

function ising_partition(sh, sv, β)
  ny, nx = size(sh)
  T = Matrix{ITensor}(undef, ny, nx)
  for iy in 1:ny, ix in 1:nx
    ixp = per(ix + 1, nx)
    iyp = per(iy + 1, ny)
    T[iy, ix] = ising_mpo(sh[iy, ix] => sh[iy, ixp], sv[iy, ix] => sv[iyp, ix], β)
  end
  return T
end

#
# Exact results
#

const βc = 0.5 * log(√2 + 1)

function ising_free_energy(β::Real, J::Real=1.0)
  k = β * J
  c = cosh(2 * k)
  s = sinh(2 * k)
  xmin = 0.0
  xmax = π
  integrand(x) = log(c^2 + √(s^4 + 1 - 2 * s^2 * cos(x)))
  integral, err = quadgk(integrand, xmin, xmax)::Tuple{Float64,Float64}
  return -(log(2) + integral / π) / (2 * β)
end

function ising_magnetization(β::Real)
  β > βc && return (1 - sinh(2 * β)^(-4))^(1 / 8)
  return 0.0
end
