
"""
A ProjMPO object represents the projection of an
MPO into a basis defined by an MPS, leaving a
certain number of site indices of the MPO unprojected.
Which sites are unprojected can be shifted by calling
the `position!` method.

Drawing of the network represented by a ProjMPO `P`, 
showing the case of `nsite(P)==2` and `position!(P,4)`:

o--o--o-      -o--o--o--o--o--o
|  |  |  |  |  |  |  |  |  |  |
o--o--o--o--o--o--o--o--o--o--o
|  |  |  |  |  |  |  |  |  |  |
o--o--o-      -o--o--o--o--o--o
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
    nsite(pm::ProjMPO)

Retrieve the number of unprojected (open)
site indices of the ProjMPO object `pm`
"""
nsite(pm::ProjMPO) = pm.nsite

"""
    length(pm::ProjMPO)

The length of a ProjMPO is the same as
the length of the MPO used to construct it
"""
Base.length(pm::ProjMPO) = length(pm.H)

function lproj(pm::ProjMPO)
  (pm.lpos <= 0) && return nothing
  return pm.LR[pm.lpos]
end

function rproj(pm::ProjMPO)
  (pm.rpos >= length(pm)+1) && return nothing
  return pm.LR[pm.rpos]
end

"""
    product(pm::ProjMPO,v::ITensor)

    (pm::ProjMPO)(v::ITensor)

Efficiently multiply the ProjMPO `pm`
by an ITensor `v` in the sense that the
ProjMPO is a generalized square matrix 
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `pm(v)` is
shorthand for `product(pm,v)`.
"""
function product(pm::ProjMPO,
                 v::ITensor)::ITensor
  Hv = v
  if isnothing(lproj(pm))
    if !isnothing(rproj(pm))
      Hv *= rproj(pm)
    end
    for j in pm.rpos-1:-1:pm.lpos+1
      Hv *= pm.H[j]
    end
  else #if lproj exists
    Hv *= lproj(pm)
    for j in pm.lpos+1:pm.rpos-1
      Hv *= pm.H[j]
    end
    if !isnothing(rproj(pm))
      Hv *= rproj(pm)
    end
  end
  return noprime(Hv)
end

(pm::ProjMPO)(v::ITensor) = product(pm,v)

"""
    eltype(pm::ProjMPO)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPO
`pm`.
"""
function Base.eltype(pm::ProjMPO)
  elT = eltype(pm.H[pm.lpos+1])
  for j in pm.lpos+2:pm.rpos-1
    elT = promote_type(elT, eltype(pm.H[j]))
  end
  if !isnothing(lproj(pm))
    elT = promote_type(elT, eltype(lproj(pm)))
  end
  if !isnothing(rproj(pm))
    elT = promote_type(elT, eltype(rproj(pm)))
  end
  return elT
end

"""
    size(pm::ProjMPO)

The size of a ProjMPO is the dimension
of the space on which it acts as a linear
operator. Thus if a ProjMPO maps from a space
with indices `(a,s1,s2,b)` to the space 
`(a',s1',s2',b')` then the size 
is `dim(a)*dim(s1)*dim(s1)*dim(b)`.
"""
function Base.size(pm::ProjMPO)::Tuple{Int,Int}
  d = 1
  if !isnothing(lproj(pm))
    for i in inds(lproj(pm))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  for j in pm.lpos+1:pm.rpos-1
    for i in inds(pm.H[j])
      plev(i) > 0 && (d *= dim(i))
    end
  end
  if !isnothing(rproj(pm))
    for i in inds(rproj(pm))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  return (d,d)
end

function makeL!(pm::ProjMPO,
                psi::MPS,
                k::Int)
  while pm.lpos < k
    ll = pm.lpos
    if ll <= 0
      pm.LR[1] = psi[1]*pm.H[1]*dag(prime(psi[1]))
      pm.lpos = 1
    else
      pm.LR[ll+1] = pm.LR[ll]*psi[ll+1]*pm.H[ll+1]*dag(prime(psi[ll+1]))
      pm.lpos += 1
    end
  end
end

function makeR!(pm::ProjMPO,
                psi::MPS,
                k::Int)
  N = length(pm.H)
  while pm.rpos > k
    rl = pm.rpos
    if rl >= N+1
      pm.LR[N] = psi[N]*pm.H[N]*dag(prime(psi[N]))
      pm.rpos = N
    else
      pm.LR[rl-1] = pm.LR[rl]*psi[rl-1]*pm.H[rl-1]*dag(prime(psi[rl-1]))
      pm.rpos -= 1
    end
  end
end

"""
    position!(pm::ProjMPO, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the ProjMPO `pm` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPO on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function position!(pm::ProjMPO,
                   psi::MPS, 
                   pos::Int)
  makeL!(pm,psi,pos-1)
  makeR!(pm,psi,pos+nsite(pm))

  #These next two lines are needed 
  #when moving lproj and rproj backward
  pm.lpos = pos-1
  pm.rpos = pos+nsite(pm)
end

# Return a "noise term" as in Phys. Rev. B 72, 180403
"""
    noiseterm(pm::ProjMPO,
              phi::ITensor,
              b::Int,
              ortho::String)
"""
function noiseterm(pm::ProjMPO,
                   phi::ITensor,
                   ortho::String)
  if nsite(pm) != 2
    error("noise term only defined for 2-site ProjMPO")
  end
  if ortho == "left"
    nt = pm.H[pm.lpos+1]*phi
    if !isnothing(lproj(pm))
      nt *= lproj(pm)
    end
  elseif ortho == "right"
    nt = phi*pm.H[pm.rpos-1]
    if !isnothing(rproj(pm))
      nt *= rproj(pm)
    end
  else
    error("In noiseterm, got ortho = $ortho, only supports `left` and `right`")
  end
  nt = nt*dag(noprime(nt))
  return nt
end

