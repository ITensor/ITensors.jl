
"""
A ProjMPO computes and stores the projection of an
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
"""
mutable struct ProjMPO <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
end
ProjMPO(H::MPO) = ProjMPO(0, length(H) + 1, 2, H, Vector{ITensor}(undef, length(H)))

copy(P::ProjMPO) = ProjMPO(P.lpos, P.rpos, P.nsite, copy(P.H), copy(P.LR))

function set_nsite!(P::ProjMPO, nsite)
  P.nsite = nsite
  return P
end
