
mutable struct ProjMPS
  lpos::Int
  rpos::Int
  nsite::Int
  M::MPS
  LR::Vector{ITensor}
end
ProjMPS(M::MPS) = ProjMPS(0, length(M) + 1, 2, M, Vector{ITensor}(undef, length(M)))

copy(P::ProjMPS) = ProjMPS(P.lpos, P.rpos, P.nsite, copy(P.M), copy(P.LR))

nsite(P::ProjMPS) = P.nsite

function set_nsite!(P::ProjMPS, nsite)
  P.nsite = nsite
  return P
end

Base.length(P::ProjMPS) = length(P.M)

function lproj(P::ProjMPS)
  (P.lpos <= 0) && return nothing
  return P.LR[P.lpos]
end

function rproj(P::ProjMPS)
  (P.rpos >= length(P) + 1) && return nothing
  return P.LR[P.rpos]
end

function product(P::ProjMPS, v::ITensor)::ITensor
  if nsite(P) != 2
    error("Only two-site ProjMPS currently supported")
  end

  Lpm = dag(prime(P.M[P.lpos + 1], "Link"))
  !isnothing(lproj(P)) && (Lpm *= lproj(P))

  Rpm = dag(prime(P.M[P.rpos - 1], "Link"))
  !isnothing(rproj(P)) && (Rpm *= rproj(P))

  pm = Lpm * Rpm

  pv = scalar(pm * v)

  Mv = pv * dag(pm)

  return noprime(Mv)
end

#function Base.eltype(P::ProjMPS)
#  elT = eltype(P.M[P.lpos+1])
#  for j = P.lpos+2:P.rpos-1
#    elT = promote_type(elT,eltype(P.M[j]))
#  end
#  if !isnull(lproj(P))
#    elT = promote_type(elT,eltype(lproj(P)))
#  end
#  if !isnull(rproj(P))
#    elT = promote_type(elT,eltype(rproj(P)))
#  end
#  return elT
#end

(P::ProjMPS)(v::ITensor) = product(P, v)

#function Base.size(P::ProjMPS)::Tuple{Int,Int}
#  d = 1
#  if P.lpos > 0
#    d *= dim(linkind(M,P.lpos))
#  end
#  for j=P.lpos+1:P.rpos-1
#    d *= dim(siteind(P.M,j))
#  end
#  if P.rpos-1 < N
#    d *= dim(linkind(M,P.rpos-1))
#  end
#  return (d,d)
#end

function makeL!(P::ProjMPS, psi::MPS, k::Int)
  while P.lpos < k
    ll = P.lpos
    if ll <= 0
      P.LR[1] = psi[1] * dag(prime(P.M[1], "Link"))
      P.lpos = 1
    else
      P.LR[ll + 1] = P.LR[ll] * psi[ll + 1] * dag(prime(P.M[ll + 1], "Link"))
      P.lpos += 1
    end
  end
end

function makeR!(P::ProjMPS, psi::MPS, k::Int)
  N = length(P.M)
  while P.rpos > k
    rl = P.rpos
    if rl >= N + 1
      P.LR[N] = psi[N] * dag(prime(P.M[N], "Link"))
      P.rpos = N
    else
      P.LR[rl - 1] = P.LR[rl] * psi[rl - 1] * dag(prime(P.M[rl - 1], "Link"))
      P.rpos -= 1
    end
  end
end

function position!(P::ProjMPS, psi::MPS, pos::Int)
  makeL!(P, psi, pos - 1)
  makeR!(P, psi, pos + nsite(P))

  #These next two lines are needed 
  #when moving lproj and rproj backward
  P.lpos = pos - 1
  P.rpos = pos + nsite(P)
  return P
end
