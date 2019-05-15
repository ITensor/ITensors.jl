using ITensors,
      LinearAlgebra,
      QuadGK

βc() = 0.5*log(sqrt(2.0)+1.0)

function ising_free_energy(β::Real,J::Real=1.0)
  k = β*J
  c = cosh(2.0*k)
  s = sinh(2.0*k)
  xmin = 0.0
  xmax = π
  integrand(x) = log(c^2+sqrt(s^4+1-2*s^2*cos(x)))
  integral,err = quadgk(integrand, xmin, xmax)::Tuple{Float64,Float64}
  return -(log(2.0)+integral/π)/(2.0*β)
end

function ising_magnetization(β::Real)
 β>βc && return (1.0-sinh(2.0*β)^(-4))^(1/8)
 return 0.0
end

function ising_mpo(sh::Tuple{Index,Index},sv::Tuple{Index,Index},
                   β::Real,J::Real=1.0;
                   sz::Bool=false)
  d = dim(sh[1])
  T = ITensor(sh[1],sh[2],sv[1],sv[2])
    for i = 1:d
      T[i,i,i,i] = 1.0
    end
    sz && (T[1,1,1,1] = -T[1,1,1,1])
    Q = [exp(β*J) exp(-β*J); exp(-β*J) exp(β*J)]
    D,U = eigen(Symmetric(Q))
    sqrtQ = U*Diagonal(sqrt.(D))*U'
    Xh1 = ITensor(vec(sqrtQ),sh[1],sh[1]')
    Xh2 = ITensor(vec(sqrtQ),sh[2],sh[2]')
    Xv1 = ITensor(vec(sqrtQ),sv[1],sv[1]')
    Xv2 = ITensor(vec(sqrtQ),sv[2],sv[2]')
    T = replacetags(T*Xh1*Xh2*Xv1*Xv2,"1","0")
  return T
end

function ising_partition(sh,sv,β)
  ny,nx = size(sh)
  T = Matrix{ITensor}(undef,ny,nx)
  for iy = 1:ny, ix = 1:nx
    ixp = per(ix+1,nx)
    iyp = per(iy+1,ny)
    T[iy,ix] = ising_mpo((sh[iy,ix],sh[iy,ixp]),(sv[iy,ix],sv[iyp,ix]),β)
  end
  return T
end

