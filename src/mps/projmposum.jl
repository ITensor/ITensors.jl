
"""
A ProjMPOSum object represents the projection of an
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

     o--o--o-      -o--o--o--o--o--o <psi|
     |  |  |  |  |  |  |  |  |  |  |
 Σⱼ  o--o--o--o--o--o--o--o--o--o--o Hⱼ
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |psi>
"""
mutable struct ProjMPOSum
  pm::Vector{ProjMPO}
end

ProjMPOSum(mpos::Vector{MPO}) = ProjMPOSum([ProjMPO(M) for M in mpos])

ProjMPOSum(Ms::MPO...) = ProjMPOSum([Ms...])

nsite(P::ProjMPOSum) = nsite(P.pm[1])

Base.length(P::ProjMPOSum) = length(P.pm[1])

"""
    product(P::ProjMPOSum,v::ITensor)

    (P::ProjMPOSum)(v::ITensor)

Efficiently multiply the ProjMPOSum `P`
by an ITensor `v` in the sense that the
ProjMPOSum is a generalized square matrix 
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.
"""
function product(P::ProjMPOSum,
                 v::ITensor)::ITensor
  Pv = product(P.pm[1],v)
  for n=2:length(P.pm)
    Pv += product(P.pm[n],v)
  end
  return Pv
end

function Base.eltype(P::ProjMPOSum)
  elT = eltype(P.pm[1])
  for n=2:length(P.pm)
    elT = promote_type(elT,eltype(P.pm[n]))
  end
  return elT
end

(P::ProjMPOSum)(v::ITensor) = product(P,v)

Base.size(P::ProjMPOSum) = size(P.pm[1])

function position!(P::ProjMPOSum,psi::MPS,pos::Int) 
  for M in P.pm
    position!(M,psi,pos)
  end
end

function noiseterm(P::ProjMPOSum,
                   phi::ITensor,
                   dir::String)
  nt = noiseterm(P.pm[1],phi,dir)
  for n=2:length(P.pm)
    nt += noiseterm(P.pm[n],phi,dir)
  end
  return nt
end
