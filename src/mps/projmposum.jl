mutable struct ProjSum{T}
  pm::Vector{T}
end

"""
A ProjMPOSum computes and stores the projection of an
implied sum of MPOs into a basis defined by an MPS,
leaving a certain number of site indices of each MPO
unprojected. Which sites are unprojected can be shifted
by calling the `position!` method. The MPOs used as
input to a ProjMPOSum are *not* added together beforehand;
instead when the `product` method of a ProjMPOSum is invoked,
each projected MPO in the set of MPOs is multiplied by
the input tensor one-by-one in an efficient way.

Drawing of the network represented by a ProjMPOSum
`P([H1,H2,...])`, showing the case of `nsite(P)==2`
and `position!(P,psi,4)` for an MPS `psi` (note the
sum Σⱼ on the left):

```
     o--o--o-      -o--o--o--o--o--o <psi|
     |  |  |  |  |  |  |  |  |  |  |
 Σⱼ  o--o--o--o--o--o--o--o--o--o--o Hⱼ
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |psi>
```
"""

copy(P::ProjSum{T}) where {T} = ProjSum{T}(copy.(P.pm))

nsite(P::ProjSum) = nsite(P.pm[1])

function set_nsite!(Ps::ProjSum, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::ProjSum) = length(P.pm[1])

function product(P::ProjSum, v::ITensor)::ITensor
  Pv = product(P.pm[1], v)
  for n in 2:length(P.pm)
    Pv += product(P.pm[n], v)
  end
  return Pv
end

function Base.eltype(P::ProjSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::ProjSum)(v::ITensor) = product(P, v)

Base.size(P::ProjSum) = size(P.pm[1])

function position!(P::ProjSum, psi::MPS, pos::Int)
  for M in P.pm
    position!(M, psi, pos)
  end
end

function noiseterm(P::ProjSum, phi::ITensor, dir::String)
  nt = noiseterm(P.pm[1], phi, dir)
  for n in 2:length(P.pm)
    nt += noiseterm(P.pm[n], phi, dir)
  end
  return nt
end

const ProjMPOSum = ProjSum{ProjMPO}

ProjMPOSum(mpos::Vector{MPO}) = ProjMPOSum([ProjMPO(M) for M in mpos])

ProjMPOSum(Ms::MPO...) = ProjMPOSum([Ms...])

const DiskProjMPOSum = ProjSum{DiskProjMPO}

function disk(ps::ProjMPOSum; kwargs...)
  return DiskProjMPOSum([disk(pm; kwargs...) for pm in ps.pm])
end
disk(P::DiskProjMPOSum; kwargs...) = P
