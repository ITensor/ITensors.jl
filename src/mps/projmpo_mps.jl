mutable struct ProjMPO_MPS
  PH::ProjMPO
  pm::Vector{ProjMPS}
  weight::Float64
end

copy(P::ProjMPO_MPS) = ProjMPO_MPS(copy(P.PH), copy.(P.pm), P.weight)

function ProjMPO_MPS(H::MPO, mpsv::Vector{MPS}; weight=1.0)
  return ProjMPO_MPS(ProjMPO(H), [ProjMPS(m) for m in mpsv], weight)
end

ProjMPO_MPS(H::MPO, Ms::MPS...; weight=1.0) = ProjMPO_MPS(H, [Ms...], weight)

nsite(P::ProjMPO_MPS) = nsite(P.PH)

function set_nsite!(Ps::ProjMPO_MPS, nsite)
  set_nsite!(Ps.PH, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::ProjMPO_MPS) = length(P.PH)

function product(P::ProjMPO_MPS, v::ITensor)::ITensor
  Pv = product(P.PH, v)
  for p in P.pm
    Pv += P.weight * product(p, v)
  end
  return Pv
end

function Base.eltype(P::ProjMPO_MPS)
  elT = eltype(P.PH)
  for p in P.pm
    elT = promote_type(elT, eltype(p))
  end
  return elT
end

(P::ProjMPO_MPS)(v::ITensor) = product(P, v)

Base.size(P::ProjMPO_MPS) = size(P.H)

function position!(P::ProjMPO_MPS, psi::MPS, pos::Int)
  position!(P.PH, psi, pos)
  for p in P.pm
    position!(p, psi, pos)
  end
  return P
end

noiseterm(P::ProjMPO_MPS, phi::ITensor, dir::String) = noiseterm(P.PH, phi, dir)
