
"""
A DiskProjMPO computes and stores the projection of an
MPO into a basis defined by an MPS, leaving a
certain number of site indices of the MPO unprojected.
Which sites are unprojected can be shifted by calling
the `position!` method.

Drawing of the network represented by a ProjMPO `P(H)`, 
showing the case of `nsite(P)==2` and `position!(P,psi,4)` 
for an MPS `psi`:

```
o--o--o-      -o--o--o--o--o--o <psi|
|  |  |  |  |  |  |  |  |  |  |
o--o--o--o--o--o--o--o--o--o--o H
|  |  |  |  |  |  |  |  |  |  |
o--o--o-      -o--o--o--o--o--o |psi>
```

The environment tensors are stored on disk, which is helpful
for large bond dimensions if they cannot fit in memory.
"""
mutable struct DiskProjMPO <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::DiskVector{ITensor}
  Lcache::Union{ITensor,Nothing}
  lposcache::Union{Int,Nothing}
  Rcache::Union{ITensor,Nothing}
  rposcache::Union{Int,Nothing}
end

DiskProjMPO(H::MPO) = new(0, length(H) + 1, 2, H, disk(Vector{ITensor}(undef, length(H))), nothing, nothing, nothing, nothing)

disk(pm::ProjMPO) = DiskProjMPO(pm.lpos, pm.rpos, pm.nsite, pm.H, disk(pm.LR), lproj(pm), pm.lpos, rproj(pm), pm.rpos)
disk(pm::DiskProjMPO) = pm

# Special overload of lproj which uses the cached
# version of the left projected MPO, and if the
# cache doesn't exist it loads it from disk.
function lproj(P::DiskProjMPO)::Union{ITensor,Nothing}
  (P.lpos <= 0) && return nothing
  if (P.lpos ≠ P.lposcache) || (P.lpos == 1)
    # Need to update the cache
    P.Lcache = P.LR[P.lpos]
    P.lposcache = P.lpos
  end
  return P.Lcache
end

# Special overload of rproj which uses the cached
# version of the right projected MPO, and if the
# cache doesn't exist it loads it from disk.
function rproj(P::DiskProjMPO)::Union{ITensor,Nothing}
  (P.rpos >= length(P) + 1) && return nothing
  if (P.rpos ≠ P.rposcache) || (P.rpos == length(P))
    # Need to update the cache
    P.Rcache = P.LR[P.rpos]
    P.rposcache = P.rpos
  end
  return P.Rcache
end

function makeL!(P::DiskProjMPO, psi::MPS, k::Int)
  _makeL!(P, psi, k)
  return P
end

function makeR!(P::DiskProjMPO, psi::MPS, k::Int)
  _makeR!(P, psi, k)
  return P
end

