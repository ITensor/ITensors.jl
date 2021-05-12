
# Scalar identity ITensor
struct OneITensor end

(::OneITensor * A::ITensor) = A
(A::ITensor * ::OneITensor) = A

abstract type AbstractProjMPO end

isnothing(::OneITensor) = error("isnothing(::OneITensor) not defined")

"""
    nsite(P::ProjMPO)

Retrieve the number of unprojected (open)
site indices of the ProjMPO object `P`
"""
nsite(P::AbstractProjMPO) = P.nsite

"""
    length(P::ProjMPO)

The length of a ProjMPO is the same as
the length of the MPO used to construct it
"""
Base.length(P::AbstractProjMPO) = length(P.H)

function lproj(P::AbstractProjMPO)
  (P.lpos <= 0) && return OneITensor()
  return P.LR[P.lpos]
end

function rproj(P::AbstractProjMPO)
  (P.rpos >= length(P) + 1) && return OneITensor()
  return P.LR[P.rpos]
end

"""
    product(P::ProjMPO,v::ITensor)::ITensor

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
function product(P::AbstractProjMPO, v::ITensor)::ITensor
  Hv = v
  L = lproj(P)
  R = rproj(P)
  # TODO: add reverse if L isa OneITensor
  site_range = (P.lpos + 1):(P.rpos - 1)
  Hv *= L
  for j in site_range
    Hv *= P.H[j]
  end
  Hv *= R
  if order(Hv) != order(v)
    error(
      string(
        "The order of the ProjMPO-ITensor product P*v is not equal to the order of the ITensor v, ",
        "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
        "(1) You are trying to multiply the ProjMPO with the $(nsite(P))-site wave-function at the wrong position.\n",
        "(2) `orthognalize!` was called, changing the MPS without updating the ProjMPO.\n\n",
        "P*v inds: $(inds(Hv)) \n\n",
        "v inds: $(inds(v))",
      ),
    )
  end
  return noprime(Hv)
end

(P::AbstractProjMPO)(v::ITensor) = product(P, v)

"""
    eltype(P::ProjMPO)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPO
`P`.
"""
function Base.eltype(P::AbstractProjMPO)
  elT = eltype(P.H[P.lpos + 1])
  for j in (P.lpos + 2):(P.rpos - 1)
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
function Base.size(P::AbstractProjMPO)::Tuple{Int,Int}
  d = 1
  if !isnothing(lproj(P))
    for i in inds(lproj(P))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  for j in (P.lpos + 1):(P.rpos - 1)
    for i in inds(P.H[j])
      plev(i) > 0 && (d *= dim(i))
    end
  end
  if !isnothing(rproj(P))
    for i in inds(rproj(P))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  return (d, d)
end

function _makeL!(P::AbstractProjMPO, psi::MPS, k::Int)::Union{ITensor,OneITensor}
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  L = OneITensor()
  while P.lpos < k
    ll = P.lpos
    if ll <= 0
      L = psi[1] * P.H[1] * dag(prime(psi[1]))
      P.LR[1] = L
      P.lpos = 1
    else
      if L isa OneITensor #isnothing(L)
        L = P.LR[ll] * psi[ll + 1] * P.H[ll + 1] * dag(prime(psi[ll + 1]))
      else
        L = L * psi[ll + 1] * P.H[ll + 1] * dag(prime(psi[ll + 1]))
      end
      P.LR[ll + 1] = L
      P.lpos += 1
    end
  end
  # Needed when moving lproj backward
  P.lpos = k
  return L
end

makeL!(P::AbstractProjMPO, psi::MPS, k::Int) = _makeL!(P, psi, k)

function _makeR!(P::AbstractProjMPO, psi::MPS, k::Int)::Union{ITensor,OneITensor}
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  R = OneITensor()
  N = length(P.H)
  while P.rpos > k
    rl = P.rpos
    if rl >= N + 1
      R = psi[N] * P.H[N] * dag(prime(psi[N]))
      P.LR[N] = R
      P.rpos = N
    else
      if R isa OneITensor #isnothing(R)
        R = P.LR[rl] * psi[rl - 1] * P.H[rl - 1] * dag(prime(psi[rl - 1]))
      else
        R = R * psi[rl - 1] * P.H[rl - 1] * dag(prime(psi[rl - 1]))
      end
      P.LR[rl - 1] = R
      P.rpos -= 1
    end
  end
  # Needed when moving rproj backward
  P.rpos = k
  return R
end

makeR!(P::AbstractProjMPO, psi::MPS, k::Int) = _makeR!(P, psi, k)

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
function position!(P::AbstractProjMPO, psi::MPS, pos::Int)
  makeL!(P, psi, pos - 1)
  makeR!(P, psi, pos + nsite(P))
  return P
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
function noiseterm(P::AbstractProjMPO, phi::ITensor, ortho::String)
  if nsite(P) != 2
    error("noise term only defined for 2-site ProjMPO")
  end

  if ortho == "left"
    AL = P.H[P.lpos + 1]
    if !isnothing(lproj(P))
      AL = lproj(P) * AL
    end
    nt = AL * phi
  elseif ortho == "right"
    AR = P.H[P.rpos - 1]
    if !isnothing(rproj(P))
      AR = AR * rproj(P)
    end
    nt = phi * AR
  else
    error("In noiseterm, got ortho = $ortho, only supports `left` and `right`")
  end
  nt = nt * dag(noprime(nt))

  return nt
end

function checkflux(P::AbstractProjMPO)
  checkflux(P.H)
  for n in length(P.LR)
    if isassigned(P.LR, n)
      checkflux(P.LR[n])
    end
  end
end
