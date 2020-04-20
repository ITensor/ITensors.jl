
mutable struct ProjMPOSum
  pm::Vector{ProjMPO}
end

ProjMPOSum(mpos::Vector{MPO}) = ProjMPOSum([ProjMPO(M) for M in mpos])

ProjMPOSum(Ms::MPO...) = ProjMPOSum([Ms...])

nsite(P::ProjMPOSum) = nsite(P.pm[1])

Base.length(P::ProjMPOSum) = length(P.pm[1])

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
                   b::Int,
                   dir::String)
  nt = noiseterm(P.pm[1],phi,b,dir)
  for n=2:length(P.pm)
    nt += noiseterm(P.pm[n],phi,b,dir)
  end
  return nt
end
