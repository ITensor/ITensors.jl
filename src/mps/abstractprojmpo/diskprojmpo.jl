
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
  Lcache::Union{ITensor,OneITensor}
  lposcache::Union{Int,Nothing}
  Rcache::Union{ITensor,OneITensor}
  rposcache::Union{Int,Nothing}
end

function copy(P::DiskProjMPO)
  return DiskProjMPO(
    P.lpos,
    P.rpos,
    P.nsite,
    copy(P.H),
    copy(P.LR),
    P.Lcache,
    P.lposcache,
    P.Rcache,
    P.rposcache,
  )
end

function set_nsite!(P::DiskProjMPO, nsite)
  P.nsite = nsite
  return P
end

function DiskProjMPO(H::MPO)
  return new(
    0,
    length(H) + 1,
    2,
    H,
    disk(Vector{ITensor}(undef, length(H))),
    OneITensor,
    nothing,
    OneITensor,
    nothing,
  )
end

function disk(pm::ProjMPO; kwargs...)
  return DiskProjMPO(
    pm.lpos,
    pm.rpos,
    pm.nsite,
    pm.H,
    disk(pm.LR; kwargs...),
    lproj(pm),
    pm.lpos,
    rproj(pm),
    pm.rpos,
  )
end
disk(pm::DiskProjMPO; kwargs...) = pm

# Special overload of lproj which uses the cached
# version of the left projected MPO, and if the
# cache doesn't exist it loads it from disk.
function lproj(P::DiskProjMPO)::Union{ITensor,OneITensor}
  (P.lpos <= 0) && return OneITensor()
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
function rproj(P::DiskProjMPO)::Union{ITensor,OneITensor}
  (P.rpos >= length(P) + 1) && return OneITensor()
  if (P.rpos ≠ P.rposcache) || (P.rpos == length(P))
    # Need to update the cache
    P.Rcache = P.LR[P.rpos]
    P.rposcache = P.rpos
  end
  return P.Rcache
end

function makeL!(P::DiskProjMPO, psi::MPS, k::Int)
  L = _makeL!(P, psi, k)
  if !isnothing(L)
    # Cache the result
    P.Lcache = L
    P.lposcache = P.lpos
  end
  return P
end

function makeR!(P::DiskProjMPO, psi::MPS, k::Int)
  R = _makeR!(P, psi, k)
  if !isnothing(R)
    # Cache the result
    P.Rcache = R
    P.rposcache = P.rpos
  end
  return P
end
