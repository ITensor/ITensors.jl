
abstract type AbstractSum end

copy(P::AbstractSum) = AbstractSum(copy.(P.terms))

nsite(P::AbstractSum) = nsite(P.terms[1])

function set_nsite!(Ps::AbstractSum, nsite)
  for P in Ps.terms
    set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::AbstractSum) = length(P.terms[1])

"""
    product(P::ProjMPOSum,v::ITensor)

    (P::ProjMPOSum)(v::ITensor)

Efficiently multiply the ProjMPOSum `P`
by an ITensor `v` in the sense that the
ProjMPOSum is a generalized square matrix
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.
"""
function product(P::AbstractSum, v::ITensor)::ITensor
  Pv = product(P.terms[1], v)
  for n in 2:length(P.terms)
    Pv += product(P.terms[n], v)
  end
  return Pv
end

"""
    eltype(P::ProjMPOSum)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPOSum
`P`.
"""
function Base.eltype(P::AbstractSum)
  elT = eltype(P.terms[1])
  for n in 2:length(P.terms)
    elT = promote_type(elT, eltype(P.terms[n]))
  end
  return elT
end

(P::AbstractSum)(v::ITensor) = product(P, v)

"""
    size(P::ProjMPOSum)

The size of a ProjMPOSum are its dimensions
`(d,d)` when viewed as a matrix or linear operator
acting on a space of dimension `d`.

For example, if a ProjMPOSum maps from a space with
indices `(a,s1,s2,b)` to the space `(a',s1',s2',b')`
then the size is `(d,d)` where
`d = dim(a)*dim(s1)*dim(s1)*dim(b)`
"""
Base.size(P::AbstractSum) = size(P.terms[1])

"""
    position!(P::ProjMPOSum, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the ProjMPOSum `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPOs on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function position!(P::AbstractSum, psi::MPS, pos::Int)
  for M in P.terms
    position!(M, psi, pos)
  end
end

"""
    noiseterm(P::ProjMPOSum,
              phi::ITensor,
              ortho::String)

Return a "noise term" or density matrix perturbation
ITensor as proposed in Phys. Rev. B 72, 180403 for aiding
convergence of DMRG calculations. The ITensor `phi`
is the contracted product of MPS tensors acted on by the
ProjMPOSum `P`, and `ortho` is a String which can take
the values `"left"` or `"right"` depending on the
sweeping direction of the DMRG calculation.
"""
function noiseterm(P::AbstractSum, phi::ITensor, dir::String)
  nt = noiseterm(P.terms[1], phi, dir)
  for n in 2:length(P.terms)
    nt += noiseterm(P.terms[n], phi, dir)
  end
  return nt
end

#
# Definition of concrete, generic ProjSum type
#

mutable struct ProjSum{T} <: AbstractSum
  terms::Vector{T}
end

ProjSum{T}(mpos::Vector{MPO}) where {T} = ProjSum([T(M) for M in mpos])

ProjSum{T}(Ms::MPO...) where {T} = ProjSum{T}([Ms...])

"""
A ProjMPOSum computes and stores the projection of an
implied sum of MPOs into a basis defined by an MPS,
leaving a certain number of site indices of each MPO
unprojected. Which sites are unprojected can be shifted
by calling the `position!` method. The MPOs used as
input to a ProjMPOSum are *not* added together beforehand;
instead when the `product` method of a ProjMPOSum is invoked,
each projected MPO in the set of MPOs is multiplied by
the input tensor one-by-one in an efficient way.

Drawing of the network represented by a ProjMPOSum
`P([H1,H2,...])`, showing the case of `nsite(P)==2`
and `position!(P,psi,4)` for an MPS `psi` (note the
sum Σⱼ on the left):

```
     o--o--o-      -o--o--o--o--o--o <psi|
     |  |  |  |  |  |  |  |  |  |  |
 Σⱼ  o--o--o--o--o--o--o--o--o--o--o Hⱼ
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |psi>
```
"""
const ProjMPOSum = ProjSum{ProjMPO}

"""
A `DiskProjMPOSum` functions the same as a `ProjMPOSum`
(see the docstring for `ProjMPOSum`) but automatically caches most of
the tensors it stores onto the hard drive to save RAM memory.
"""
const DiskProjMPOSum = ProjSum{DiskProjMPO}

"""
    disk(ps::ProjMPOSum; kwargs...)

Convert a ProjMPOSum into a DiskProjMPOSum,
which will automatically start caching most 
stored tensors onto the hard drive.
"""
function disk(ps::ProjMPOSum; kwargs...)
  return DiskProjMPOSum([disk(pm; kwargs...) for pm in ps.terms])
end

disk(P::DiskProjMPOSum; kwargs...) = P
