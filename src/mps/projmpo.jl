
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
mutable struct ProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  H::MPO
  LR::Vector{ITensor}
  ProjMPO(H::MPO) = new(0,length(H)+1,2,H,
                        Vector{ITensor}(undef, length(H)))
end

"""
    nsite(P::ProjMPO)

Retrieve the number of unprojected (open)
site indices of the ProjMPO object `P`
"""
nsite(P::ProjMPO) = P.nsite

"""
    length(P::ProjMPO)

The length of a ProjMPO is the same as
the length of the MPO used to construct it
"""
Base.length(P::ProjMPO) = length(P.H)

function lproj(P::ProjMPO)
  (P.lpos <= 0) && return nothing
  return P.LR[P.lpos]
end

function rproj(P::ProjMPO)
  (P.rpos >= length(P)+1) && return nothing
  return P.LR[P.rpos]
end

"""
    product(P::ProjMPO,v::ITensor)

    (P::ProjMPO)(v::ITensor)

Efficiently multiply the ProjMPO `P`
by an ITensor `v` in the sense that the
ProjMPO is a generalized square matrix 
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.
"""
function product(P::ProjMPO,
                 v::ITensor)::ITensor
  Hv = v
  if isnothing(lproj(P))
    if !isnothing(rproj(P))
      Hv *= rproj(P)
    end
    for j in P.rpos-1:-1:P.lpos+1
      Hv *= P.H[j]
    end
  else #if lproj exists
    Hv *= lproj(P)
    for j in P.lpos+1:P.rpos-1
      Hv *= P.H[j]
    end
    if !isnothing(rproj(P))
      Hv *= rproj(P)
    end
  end
  return noprime(Hv)
end

(P::ProjMPO)(v::ITensor) = product(P,v)

"""
    eltype(P::ProjMPO)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPO
`P`.
"""
function Base.eltype(P::ProjMPO)
  elT = eltype(P.H[P.lpos+1])
  for j in P.lpos+2:P.rpos-1
    elT = promote_type(elT, eltype(P.H[j]))
  end
  if !isnothing(lproj(P))
    elT = promote_type(elT, eltype(lproj(P)))
  end
  if !isnothing(rproj(P))
    elT = promote_type(elT, eltype(rproj(P)))
  end
  return elT
end

"""
    size(P::ProjMPO)

The size of a ProjMPO are its dimensions
`(d,d)` when viewed as a matrix or linear operator
acting on a space of dimension `d`. 

For example, if a ProjMPO maps from a space with 
indices `(a,s1,s2,b)` to the space `(a',s1',s2',b')` 
then the size is `(d,d)` where 
`d = dim(a)*dim(s1)*dim(s1)*dim(b)`
"""
function Base.size(P::ProjMPO)::Tuple{Int,Int}
  d = 1
  if !isnothing(lproj(P))
    for i in inds(lproj(P))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  for j in P.lpos+1:P.rpos-1
    for i in inds(P.H[j])
      plev(i) > 0 && (d *= dim(i))
    end
  end
  if !isnothing(rproj(P))
    for i in inds(rproj(P))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  return (d,d)
end

function makeL!(P::ProjMPO,
                psi::MPS,
                k::Int)
  while P.lpos < k
    ll = P.lpos
    if ll <= 0
      P.LR[1] = psi[1]*P.H[1]*dag(prime(psi[1]))
      P.lpos = 1
    else
      P.LR[ll+1] = P.LR[ll]*psi[ll+1]*P.H[ll+1]*dag(prime(psi[ll+1]))
      P.lpos += 1
    end
  end
end

function makeR!(P::ProjMPO,
                psi::MPS,
                k::Int)
  N = length(P.H)
  while P.rpos > k
    rl = P.rpos
    if rl >= N+1
      P.LR[N] = psi[N]*P.H[N]*dag(prime(psi[N]))
      P.rpos = N
    else
      P.LR[rl-1] = P.LR[rl]*psi[rl-1]*P.H[rl-1]*dag(prime(psi[rl-1]))
      P.rpos -= 1
    end
  end
end

"""
    position!(P::ProjMPO, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the ProjMPO `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPO on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function position!(P::ProjMPO,
                   psi::MPS, 
                   pos::Int)
  makeL!(P,psi,pos-1)
  makeR!(P,psi,pos+nsite(P))

  #These next two lines are needed 
  #when moving lproj and rproj backward
  P.lpos = pos-1
  P.rpos = pos+nsite(P)
end

"""
    noiseterm(P::ProjMPO,
              phi::ITensor,
              ortho::String)

Return a "noise term" or density matrix perturbation
ITensor as proposed in Phys. Rev. B 72, 180403 for aiding
convergence of DMRG calculations. The ITensor `phi`
is the contracted product of MPS tensors acted on by the 
ProjMPO `P`, and `ortho` is a String which can take
the values `"left"` or `"right"` depending on the 
sweeping direction of the DMRG calculation.
"""
function noiseterm(P::ProjMPO,
                   phi::ITensor,
                   ortho::String)
  if nsite(P) != 2
    error("noise term only defined for 2-site ProjMPO")
  end
  if ortho == "left"
    nt = P.H[P.lpos+1]*phi
    if !isnothing(lproj(P))
      nt *= lproj(P)
    end
  elseif ortho == "right"
    nt = phi*P.H[P.rpos-1]
    if !isnothing(rproj(P))
      nt *= rproj(P)
    end
  else
    error("In noiseterm, got ortho = $ortho, only supports `left` and `right`")
  end
  nt = nt*dag(noprime(nt))
  return nt
end

