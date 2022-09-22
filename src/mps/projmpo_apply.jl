
"""
```
     o--o--o-      -o--o--o--o--o--o <ΨB|
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o--o--o--o--o--o--o--o--o H  
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |ΨA>
```
"""
mutable struct ProjMPOApply <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  psi0::MPS
  H::MPO
  LR::Vector{ITensor}
end

function ProjMPOApply(psi0::MPS, H::MPO)
  return ProjMPO(0, length(H) + 1, 2, psi0, H, Vector{ITensor}(undef, length(H)))
end

function copy(P::ProjMPOApply)
  return ProjMPOApply(P.lpos, P.rpos, P.nsite, copy(P.psi0), copy(P.H), copy(P.LR))
end

function set_nsite!(P::ProjMPOApply, nsite)
  P.nsite = nsite
  return P
end

function makeL!(P::ProjMPOApply, psi::MPS, k::Int)
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
    L = L * P.psi0[ll + 1] * P.H[ll + 1] * dag(prime(psi[ll + 1]))
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return P
end

function makeR!(P::ProjMPOApply, psi::MPS, k::Int)
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
  N = length(P.H)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(P)
  while rl > k
    R = R * P.psi0[rl - 1] * P.H[rl - 1] * dag(prime(psi[rl - 1]))
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return P
end
