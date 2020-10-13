using ITensors
using LinearAlgebra
using QuadGK

function ising_mpo(sh::Tuple{Index, Index},
                   sv::Tuple{Index, Index},
                   β::Real, J::Real = 1.0;
                   sz::Bool = false)
  d = dim(sh[1])
  T = ITensor(sh[1], sh[2], sv[1], sv[2])
  for i in 1:d
    T[i, i, i, i] = 1
  end
  if sz
    T[1, 1, 1, 1] = -T[1, 1, 1, 1]
  end
  Q = [exp(β * J) exp(-β * J); exp(-β * J) exp(β * J)]
  D,U = eigen(Symmetric(Q))
  √Q = U * Diagonal(sqrt.(D)) * U'
  Xh1 = itensor(vec(√Q), sh[1], sh[1]')
  Xh2 = itensor(vec(√Q), sh[2], sh[2]')
  Xv1 = itensor(vec(√Q), sv[1], sv[1]')
  Xv2 = itensor(vec(√Q), sv[2], sv[2]')
  T = mapprime(T * Xh1 * Xh2 * Xv1 * Xv2, 1 => 0)
  return T
end

function ising_mpo_dual(sh::Tuple{Index, Index},
                        sv::Tuple{Index, Index},
                        β::Real, J::Real = 1.0)
  d = dim(sh[1])
  T = ITensor(sh[1], sh[2], sv[1], sv[2])
  sig(s) = 1.0 - 2.0 * (s - 1)
  E0 = -4.0
  for s1 in 1:d, s2 in 1:d, s3 in 1:d, s4 in 1:d
    E = sig(s1) * sig(s2) +
        sig(s2) * sig(s3) +
        sig(s3) * sig(s4) +
        sig(s4) * sig(s1)
    val = exp(-β * (E - E0))
    T[sh[1](s1), sv[2](s2), sh[2](s3), sv[1](s4)] = val
  end
  return T
end

function ising_partition(sh, sv, β)
  ny,nx = size(sh)
  T = Matrix{ITensor}(undef, ny, nx)
  for iy in 1:ny, ix in 1:nx
    ixp = per(ix + 1, nx)
    iyp = per(iy + 1, ny)
    T[iy, ix] = ising_mpo((sh[iy, ix], sh[iy, ixp]),
                          (sv[iy, ix], sv[iyp, ix]), β)
  end
  return T
end

#
# Exact results
#

const βc = 0.5 * log(√2 + 1)

function ising_free_energy(β::Real, J::Real = 1.0)
  k = β * J
  c = cosh(2 * k)
  s = sinh(2 * k)
  xmin = 0.0
  xmax = π
  integrand(x) = log(c^2 + √(s^4 + 1 - 2 * s^2 * cos(x)))
  integral, err = quadgk(integrand, xmin, xmax)::Tuple{Float64, Float64}
  return -(log(2) + integral / π) / (2 * β)
end

function ising_magnetization(β::Real)
 β > βc && return (1 - sinh(2 * β)^(-4))^(1 / 8)
 return 0.0
end

