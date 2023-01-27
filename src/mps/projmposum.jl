
abstract type AbstractSum end

terms(sum::AbstractSum) = sum.terms

function set_terms(sum::AbstractSum, terms)
  return error("Please implement `set_terms` for the `AbstractSum` type `$(typeof(sum))`.")
end

copy(P::AbstractSum) = typeof(P)(copy.(terms(P)))

nsite(P::AbstractSum) = nsite(first(terms(P)))

function set_nsite!(A::AbstractSum, nsite)
  for t in terms(A)
    set_nsite!(t, nsite)
  end
  return A
end

length(A::AbstractSum) = length(terms(A)[1])

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
function product(A::AbstractSum, v::ITensor)::ITensor
  Av = product(first(terms(A)), v)
  for n in 2:length(terms(A))
    Av += product(terms(A)[n], v)
  end
  return Av
end

"""
    eltype(P::ProjMPOSum)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPOSum
`P`.
"""
function eltype(A::AbstractSum)
  elT = eltype(first(terms(A)))
  for n in 2:length(terms(A))
    elT = promote_type(elT, eltype(terms(A)[n]))
  end
  return elT
end

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
size(A::AbstractSum) = size(first(terms(A)))

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
  for t in terms(A)
    position!(t, psi, pos)
  end
  return A
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
  nt = noiseterm(terms(A)[1], phi, dir)
  for n in 2:length(terms(A))
    nt += noiseterm(terms(A)[n], phi, dir)
  end
  return nt
end

"""
    disk(ps::AbstractSum; kwargs...)

Call `disk` on each term of an AbstractSum, to enable
saving of cached data to hard disk.
"""
function disk(sum::AbstractSum; disk_kwargs...)
  return set_terms(sum, [disk(term; disk_kwargs...) for term in terms(sum)])
end

#
# Definition of concrete, generic SequentialSum type
#

struct SequentialSum{T} <: AbstractSum
  terms::Vector{T}
end

SequentialSum{T}(mpos::Vector{MPO}) where {T} = SequentialSum([T(M) for M in mpos])

SequentialSum{T}(Ms::MPO...) where {T} = SequentialSum{T}([Ms...])

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
