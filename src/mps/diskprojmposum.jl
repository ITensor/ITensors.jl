
mutable struct DiskProjMPOSum
  pm::Vector{DiskProjMPO}
end

copy(P::DiskProjMPOSum) = DiskProjMPOSum(copy.(P.pm))

DiskProjMPOSum(mpos::Vector{MPO}) = DiskProjMPOSum([ProjMPO(M) for M in mpos])

DiskProjMPOSum(Ms::MPO...) = DiskProjMPOSum([Ms...])

nsite(P::DiskProjMPOSum) = nsite(P.pm[1])

function set_nsite!(Ps::DiskProjMPOSum, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::DiskProjMPOSum) = length(P.pm[1])

"""
    product(P::DiskProjMPOSum,v::ITensor)

    (P::DiskProjMPOSum)(v::ITensor)

Efficiently multiply the DiskProjMPOSum `P`
by an ITensor `v` in the sense that the
DiskProjMPOSum is a generalized square matrix
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.
"""
function product(P::DiskProjMPOSum, v::ITensor)::ITensor
  Pv = product(P.pm[1], v)
  for n in 2:length(P.pm)
    Pv += product(P.pm[n], v)
  end
  return Pv
end

"""
    eltype(P::DiskProjMPOSum)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the DiskProjMPOSum
`P`.
"""
function Base.eltype(P::DiskProjMPOSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::DiskProjMPOSum)(v::ITensor) = product(P, v)

"""
    size(P::DiskProjMPOSum)

The size of a DiskProjMPOSum are its dimensions
`(d,d)` when viewed as a matrix or linear operator
acting on a space of dimension `d`.

For example, if a DiskProjMPOSum maps from a space with
indices `(a,s1,s2,b)` to the space `(a',s1',s2',b')`
then the size is `(d,d)` where
`d = dim(a)*dim(s1)*dim(s1)*dim(b)`
"""
Base.size(P::DiskProjMPOSum) = size(P.pm[1])

"""
    position!(P::DiskProjMPOSum, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the DiskProjMPOSum `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPOs on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function ITensors.position!(P::DiskProjMPOSum, psi::MPS, pos::Int)
  for M in P.pm
    position!(M, psi, pos)
  end
end

"""
    noiseterm(P::DiskProjMPOSum,
              phi::ITensor,
              ortho::String)

Return a "noise term" or density matrix perturbation
ITensor as proposed in Phys. Rev. B 72, 180403 for aiding
convergence of DMRG calculations. The ITensor `phi`
is the contracted product of MPS tensors acted on by the
DiskProjMPOSum `P`, and `ortho` is a String which can take
the values `"left"` or `"right"` depending on the
sweeping direction of the DMRG calculation.
"""
function noiseterm(P::DiskProjMPOSum, phi::ITensor, dir::String)
  nt = noiseterm(P.pm[1], phi, dir)
  for n in 2:length(P.pm)
    nt += noiseterm(P.pm[n], phi, dir)
  end
  return nt
end

function disk(ps::ProjMPOSum; kwargs...)
  return DiskProjMPOSum([disk(pm; kwargs...) for pm in ps.pm])
end
disk(P::DiskProjMPOSum; kwargs...) = P
