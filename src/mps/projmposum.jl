
abstract type AbstractSum end

terms(sum::AbstractSum) = sum.terms

function set_terms(sum::AbstractSum, terms)
  return error("Please implement `set_terms` for the `AbstractSum` type `$(typeof(sum))`.")
end

copy(P::AbstractSum) = typeof(P)(copy.(terms(P)))

function nsite(P::AbstractSum)
  @assert allequal(nsite.(terms(P)))
  return nsite(first(terms(P)))
end

function set_nsite!(A::AbstractSum, nsite)
  return set_terms(A, map(term -> set_nsite!(term, nsite), terms(A)))
end

function length(A::AbstractSum)
  @assert allequal(length.(terms(A)))
  return length(first(terms(A)))
end

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
product(A::AbstractSum, v::ITensor) = sum(t -> product(t, v), terms(A))

"""
    eltype(P::ProjMPOSum)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPOSum
`P`.
"""
eltype(A::AbstractSum) = mapreduce(eltype, promote_type, terms(A))

(A::AbstractSum)(v::ITensor) = product(A, v)

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
function size(A::AbstractSum)
  @assert allequal(size.(terms(A)))
  return size(first(terms(A)))
end

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
function position!(A::AbstractSum, psi::MPS, pos::Int)
  new_terms = map(term -> position!(term, psi, pos), terms(A))
  return set_terms(A, new_terms)
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
function noiseterm(A::AbstractSum, phi::ITensor, dir::String)
  return sum(t -> noiseterm(t, phi, dir), terms(A))
end

"""
    disk(ps::AbstractSum; kwargs...)

Call `disk` on each term of an AbstractSum, to enable
saving of cached data to hard disk.
"""
function disk(sum::AbstractSum; disk_kwargs...)
  return set_terms(sum, map(t -> disk(t; disk_kwargs...), terms(sum)))
end

#
# Definition of concrete, generic SequentialSum type
#

struct SequentialSum{T} <: AbstractSum
  terms::Vector{T}
end

function SequentialSum{T}(mpos::Vector{MPO}) where {T<:AbstractProjMPO}
  return SequentialSum([T(M) for M in mpos])
end

SequentialSum{T}(Ms::MPO...) where {T<:AbstractProjMPO} = SequentialSum{T}([Ms...])

set_terms(sum::SequentialSum, terms) = SequentialSum(terms)

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
const ProjMPOSum = SequentialSum{ProjMPO}
