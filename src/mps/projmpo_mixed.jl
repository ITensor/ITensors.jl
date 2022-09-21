
"""
```
     o--o--o-      -o--o--o--o--o--o <ΨB|
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o--o--o--o--o--o--o--o--o H  
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |ΨA>
```
"""
mutable struct ProjMPOMixed <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
end

function makeL!(P::ProjMPOMixed, psiA::MPS, psiB::MPS, k::Int)
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
    L = L * psiA[ll + 1] * P.H[ll + 1] * dag(prime(psiB[ll + 1]))
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return P
end

function makeR!(P::ProjMPOMixed, psiA::MPS, psiB::MPS, k::Int)
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
    R = R * psiA[rl - 1] * P.H[rl - 1] * dag(prime(psiB[rl - 1]))
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return P
end

function position!(P::ProjMPOMixed, psiA::MPS, psiB::MPS, pos::Int)
  makeL!(P, psiA, psiB, pos - 1)
  makeR!(P, psiA, psiB, pos + nsite(P))
  return P
end
