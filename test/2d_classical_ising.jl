using ITensors
import QuadGK

const βc = 0.5*log(sqrt(2.0)+1.0)

function ising_free_energy(β::Real,J::Real=1.0)
  k = β*J
  c = cosh(2.0*k)
  s = sinh(2.0*k)
  xmin = 0.0
  xmax = π
  integrand(x) = log(c^2+sqrt(s^4+1-2*s^2*cos(x)))
  integral,err = QuadGK.quadgk(integrand, xmin, xmax)::Tuple{Float64,Float64}
  return -(log(2.0)+integral/π)/(2.0*β)
end

function ising_magnetization(β::Real)
 β>βc && return (1.0-sinh(2.0*β)^(-4))^(1/8)
 return 0.0
end

function ising_mpo(sh::Tuple{Index,Index},sv::Tuple{Index,Index},
                   β::Real,J::Real=1.0;
                   sz::Bool=false,dual_lattice::Bool=true)
  d = dim(sh[1])
  T = ITensor(sh[1],sh[2],sv[1],sv[2])
  if dual_lattice
    for i = 1:d
      T[i,i,i,i] = 1.0
    end
    sz && (T[1,1,1,1] = -T[1,1,1,1])
    Q = [exp(β*J) exp(-β*J); exp(-β*J) exp(β*J)]
    D,U = eigen(LinearAlgebra.Symmetric(Q))
    √Q = U*LinearAlgebra.Diagonal(sqrt.(D))*U'
    Xh1 = itensor(√Q,sh[1],sh[1]')
    Xh2 = itensor(√Q,sh[2],sh[2]')
    Xv1 = itensor(√Q,sv[1],sv[1]')
    Xv2 = itensor(√Q,sv[2],sv[2]')
    T = noprime(T*Xh1*Xh2*Xv1*Xv2)
  else
    sig(s) = 1.0-2.0*(s-1)
    E0 = -4.0
    for s1 = 1:d, s2 = 1:d, s3 = 1:d, s4 = 1:d
      E = sig(s1)*sig(s2)+sig(s2)*sig(s3)+sig(s3)*sig(s4)+sig(s4)*sig(s1)
      val = exp(-β*(E-E0))
      T[sh[1](s1),sv[2](s2),sh[2](s3),sv[1](s4)] = val
    end
  end
  return T
end

