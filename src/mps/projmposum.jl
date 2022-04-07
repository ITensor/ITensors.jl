
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
mutable struct ProjMPOSum
  pm::Vector{ProjMPO}
end

copy(P::ProjMPOSum) = ProjMPOSum(copy.(P.pm))

ProjMPOSum(mpos::Vector{MPO}) = ProjMPOSum([ProjMPO(M) for M in mpos])

ProjMPOSum(Ms::MPO...) = ProjMPOSum([Ms...])

nsite(P::ProjMPOSum) = nsite(P.pm[1])

function set_nsite!(Ps::ProjMPOSum, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::ProjMPOSum) = length(P.pm[1])

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
function product(P::ProjMPOSum, v::ITensor)::ITensor
  Pv = product(P.pm[1], v)
  for n in 2:length(P.pm)
    Pv += product(P.pm[n], v)
  end
  return Pv
end

"""
    eltype(P::ProjMPOSum)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPOSum
`P`.
"""
function Base.eltype(P::ProjMPOSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::ProjMPOSum)(v::ITensor) = product(P, v)

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
Base.size(P::ProjMPOSum) = size(P.pm[1])

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
function position!(P::ProjMPOSum, psi::MPS, pos::Int)
  for M in P.pm
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
function noiseterm(P::ProjMPOSum, phi::ITensor, dir::String)
  nt = noiseterm(P.pm[1], phi, dir)
  for n in 2:length(P.pm)
    nt += noiseterm(P.pm[n], phi, dir)
  end
  return nt
end
