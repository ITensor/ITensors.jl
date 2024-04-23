using ITensors: ITensors, ITensor, dag, dim, prime

"""
Holds the following data where psi
is the MPS being optimized and M is the 
MPS held constant by the ProjMPS.
```
     o--o--o--o--o--o--o--o--o--o--o <M|
     |  |  |  |  |  |  |  |  |  |  |
     *--*--*-      -*--*--*--*--*--* |psi>
```
"""
mutable struct ProjMPS2 <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  M::MPS
  LR::Vector{ITensor}
end

function ProjMPS2(M::MPS)
  return ProjMPS2(0, length(M) + 1, 2, M, Vector{ITensor}(undef, length(M)))
end

Base.length(P::ProjMPS2) = length(P.M)

function Base.copy(P::ProjMPS2)
  return ProjMPS2(P.lpos, P.rpos, P.nsite, copy(P.M), copy(P.LR))
end

function set_nsite!(P::ProjMPS2, nsite)
  P.nsite = nsite
  return P
end

function makeL!(P::ProjMPS2, psi::MPS, k::Int)
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  ll = P.lpos
  if ll ≥ k
    # Special case when nothing has to be done.
    # Still need to change the position if lproj is
    # being moved backward.
    P.lpos = k
    return nothing
  end
  # Make sure ll is at least 0 for the generic logic below
  ll = max(ll, 0)
  L = lproj(P)
  while ll < k
    L = L * psi[ll + 1] * dag(prime(P.M[ll + 1], "Link"))
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return P
end

function makeR!(P::ProjMPS2, psi::MPS, k::Int)
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  rl = P.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    P.rpos = k
    return nothing
  end
  N = length(P.M)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(P)
  while rl > k
    R = R * psi[rl - 1] * dag(prime(P.M[rl - 1], "Link"))
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return P
end

function ITensors.contract(P::ProjMPS2, v::ITensor)
  itensor_map = Union{ITensor,OneITensor}[lproj(P)]
  append!(itensor_map, [prime(t, "Link") for t in P.M[site_range(P)]])
  push!(itensor_map, rproj(P))

  # Reverse the contraction order of the map if
  # the first tensor is a scalar (for example we
  # are at the left edge of the system)
  if dim(first(itensor_map)) == 1
    reverse!(itensor_map)
  end

  # Apply the map
  Mv = v
  for it in itensor_map
    Mv *= it
  end
  return Mv
end

function proj_mps(P::ProjMPS2)
  itensor_map = Union{ITensor,OneITensor}[lproj(P)]
  append!(itensor_map, [dag(prime(t, "Link")) for t in P.M[site_range(P)]])
  push!(itensor_map, rproj(P))

  # Reverse the contraction order of the map if
  # the first tensor is a scalar (for example we
  # are at the left edge of the system)
  if dim(first(itensor_map)) == 1
    reverse!(itensor_map)
  end

  # Apply the map
  m = ITensor(true)
  for it in itensor_map
    m *= it
  end
  return m
end
