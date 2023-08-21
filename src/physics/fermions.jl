
"""
    parity_sign(P)

Given an array or tuple of integers representing
a permutation or a subset of a permutation,
compute the parity sign defined as -1 for a
permutation consisting of an odd number of swaps
and +1 for an even number of swaps. This
implementation uses an O(n^2) algorithm and is
intended for small permutations only.
"""
function parity_sign(P)::Int
  L = length(P)
  s = +1
  for i in 1:L, j in (i + 1):L
    s *= sign(P[j] - P[i])
  end
  return s
end

isfermionic(qv::QNVal) = (modulus(qv) < 0)

isfermionic(qn::QN) = any(isfermionic, qn)

has_fermionic_subspaces(i::Index) = false

function has_fermionic_subspaces(i::QNIndex)
  for b in 1:nblocks(i)
    isfermionic(qn(i, b)) && (return true)
  end
  return false
end

isfermionic(i::Index) = has_fermionic_subspaces(i)

has_fermionic_subspaces(is::Indices) = false

function has_fermionic_subspaces(is::QNIndices)
  for i in is, b in 1:nblocks(i)
    isfermionic(qn(i, b)) && (return true)
  end
  return false
end

has_fermionic_subspaces(T::Tensor) = has_fermionic_subspaces(inds(T))

"""
    fparity(qn::QN)
    fparity(qn::IndexVal)

Compute the fermion parity (0 or 1) of a QN of IndexVal,
defined as the sum mod 2 of each of its fermionic
QNVals (QNVals with negative modulus).
"""
function fparity(qn::QN)
  p = 0
  for qv in qn
    if isfermionic(qv)
      p += val(qv)
    end
  end
  return mod(p, 2)
end

fparity(iv::Pair{<:Index}) = fparity(qn(iv))

Base.isodd(q::QN) = isodd(fparity(q))
Base.isodd(iv::Pair{<:Index}) = isodd(fparity(iv))

"""
    compute_permfactor(p,iv_or_qn::Vararg{T,N})

Given a permutation p and a set "s" of QNIndexVals or QNs,
if the subset of index vals which are fermion-parity
odd undergo an odd permutation (odd number of swaps)
according to p, then return -1. Otherwise return +1.
"""
function compute_permfactor(p, iv_or_qn...; range=1:length(iv_or_qn))::Int
  !using_auto_fermion() && return 1
  N = length(iv_or_qn)
  # XXX: Bug https://github.com/ITensor/ITensors.jl/issues/931
  # oddp = @MVector zeros(Int, N)
  oddp = MVector((ntuple(Returns(0), Val(N))))
  n = 0
  @inbounds for j in range
    if fparity(iv_or_qn[p[j]]) == 1
      n += 1
      oddp[n] = p[j]
    end
  end
  return parity_sign(oddp[1:n])
end

function NDTensors.permfactor(p, ivs::Vararg{Pair{QNIndex},N}; kwargs...) where {N}
  !using_auto_fermion() && return 1
  return compute_permfactor(p, ivs...; kwargs...)
end

function NDTensors.permfactor(
  perm, block::NDTensors.Block{N}, inds::QNIndices; kwargs...
) where {N}
  !using_auto_fermion() && return 1
  qns = ntuple(n -> qn(inds[n], block[n]), N)
  return compute_permfactor(perm, qns...; kwargs...)
end

NDTensors.block_parity(i::QNIndex, block::Integer) = fparity(qn(i, block))

function NDTensors.right_arrow_sign(i::QNIndex, block::Integer)
  !using_auto_fermion() && return 1
  if dir(i) == Out && NDTensors.block_parity(i, block) == 1
    return -1
  end
  return 1
end

function NDTensors.left_arrow_sign(i::QNIndex, block::Integer)
  !using_auto_fermion() && return 1
  if dir(i) == In && NDTensors.block_parity(i, block) == 1
    return -1
  end
  return 1
end

# Version of getperm which is type stable
# and works for Tuple or Vector inputs
function vec_getperm(s1, s2)
  N = length(s1)
  p = Vector{Int}(undef, N)
  for i in 1:N
    @inbounds p[i] = NDTensors._findfirst(==(@inbounds s1[i]), s2)
  end
  return p
end

@inline function NDTensors.compute_alpha(
  ElR,
  labelsR,
  blockR,
  input_indsR,
  labelsT1,
  blockT1,
  indsT1::NTuple{N1,QNIndex},
  labelsT2,
  blockT2,
  indsT2::NTuple{N2,QNIndex},
) where {N1,N2}
  if !using_auto_fermion()
    !has_fermionic_subspaces(indsT1) || !has_fermionic_subspaces(indsT2)
    return one(ElR)
  end

  # the "indsR" argument to compute_alpha from NDTensors
  # may be a tuple of QNIndex, so convert to a Vector{Index}
  indsR = collect(input_indsR)

  nlabelsT1 = NDTensors.sort(labelsT1; rev=true)
  nlabelsT2 = NDTensors.sort(labelsT2)

  # Make orig_labelsR from the order of
  # indices that would result by just 
  # taking the uncontracted indices of
  # T1 and T2 in their input order:
  NR = length(labelsR)
  orig_labelsR = zeros(Int, NR)
  u = 1
  for ls in (nlabelsT1, nlabelsT2), l in ls
    if l > 0
      orig_labelsR[u] = l
      u += 1
    end
  end

  permT1 = NDTensors.getperm(nlabelsT1, labelsT1)
  permT2 = NDTensors.getperm(nlabelsT2, labelsT2)
  permR = vec_getperm(labelsR, orig_labelsR)

  alpha1 = NDTensors.permfactor(permT1, blockT1, indsT1)
  alpha2 = NDTensors.permfactor(permT2, blockT2, indsT2)
  alphaR = NDTensors.permfactor(permR, blockR, indsR)

  alpha_arrows = one(ElR)
  for n in 1:length(indsT1)
    l = labelsT1[n]
    i = indsT1[n]
    qi = qn(i, blockT1[n])
    if l < 0 && dir(i) == Out && fparity(qi) == 1
      alpha_arrows *= -1
    end
  end

  α = alpha1 * alpha2 * alphaR * alpha_arrows

  return α
end

# Flip signs of selected blocks of T prior to
# it being multiplied by a combiner ITensor
# labelsR gives the ordering of indices after the product
function NDTensors.before_combiner_signs(
  T,
  labelsT_,
  indsT::NTuple{NT,QNIndex},
  C,
  labelsC_,
  indsC::NTuple{NC,QNIndex},
  labelsR,
  indsR::NTuple{NR,QNIndex},
) where {NC,NT,NR}
  if !using_auto_fermion() || !has_fermionic_subspaces(T)
    return T
  end

  T = copy(T)

  labelsC = [l for l in labelsC_]
  labelsT = [l for l in labelsT_]

  # number of uncombined indices
  Nuc = NC - 1

  ci = NDTensors.cinds(storage(C))[1]
  combining = (labelsC[ci] > 0)

  isconj = NDTensors.isconj(storage(C))

  if combining
    #println("Combining <<<<<<<<<<<<<<<<<<<<<<<<<<<")

    nlabelsT = Int[]

    if !isconj
      # Permute uncombined indices to front
      # in same order as indices passed to the
      # combiner constructor
      append!(nlabelsT, labelsC[2:end])
    else # isconj
      # If combiner is conjugated, put uncombined
      # indices in *opposite* order as on combiner
      append!(nlabelsT, reverse(labelsC[2:end]))
    end
    @assert all(l -> l < 0, nlabelsT)

    for l in labelsT
      if l > 0 #uncontracted
        append!(nlabelsT, l)
      end
    end
    @assert length(nlabelsT) == NT

    # Compute permutation that moves uncombined indices to front
    permT = vec_getperm(nlabelsT, labelsT)

    for blockT in keys(blockoffsets(T))
      # Compute sign from permuting uncombined indices to front:
      alphaT = NDTensors.permfactor(permT, blockT, indsT)

      neg_dir = !isconj ? In : Out
      alpha_arrows = 1
      alpha_mixed_arrow = 1
      C_dir = dir(indsC[1])
      for n in 1:length(indsT)
        i = indsT[n]
        qi = qn(i, blockT[n])
        if labelsT[n] < 0 && fparity(qi) == 1
          alpha_mixed_arrow *= (dir(i) != C_dir) ? -1 : +1
          alpha_arrows *= (dir(i) == neg_dir) ? -1 : +1
        end
      end

      fac = alphaT * alpha_arrows

      if isconj
        fac *= alpha_mixed_arrow
      end

      if fac != 1
        Tb = blockview(T, blockT)
        scale!(Tb, fac)
      end
    end # for blockT

  elseif !combining
    #
    # Uncombining ---------------------------
    #
    #println("Uncombining >>>>>>>>>>>>>>>>>>>>>>>>>>>")

    nc = findfirst(l -> l < 0, labelsT)
    nlabelsT = [labelsT[nc]]
    ic = indsT[nc]

    for l in labelsT
      (l > 0) && append!(nlabelsT, l)
    end

    # Compute sign for permuting combined index to front
    # (sign alphaT to be computed for each block below):
    permT = vec_getperm(nlabelsT, labelsT)

    #
    # Note: other permutation of labelsT which
    # relates to two treatments of isconj==true/false
    # in combining case above is handled as a 
    # post-processing step in NDTensors.after_combiner_signs
    # implemented below
    #

    for blockT in keys(blockoffsets(T))
      alphaT = NDTensors.permfactor(permT, blockT, indsT)

      neg_dir = !isconj ? Out : In
      qic = qn(ic, blockT[nc])
      alpha_arrows = (fparity(qic) == 1 && dir(ic) == neg_dir) ? -1 : +1

      fac = alphaT * alpha_arrows

      if fac != 1
        Tb = blockview(T, blockT)
        scale!(Tb, fac)
      end
    end
  end

  return T
end #NDTensors.before_combiner_signs

function NDTensors.after_combiner_signs(
  R, labelsR, indsR::NTuple{NR,QNIndex}, C, labelsC, indsC::NTuple{NC,QNIndex}
) where {NC,NR}
  ci = NDTensors.cinds(store(C))[1]
  combining = (labelsC[ci] > 0)
  combining && error("NDTensors.after_combiner_signs only for uncombining")

  if !using_auto_fermion() || !has_fermionic_subspaces(R)
    return R
  end

  R = copy(R)

  # number of uncombined indices
  Nuc = NC - 1

  isconj = NDTensors.isconj(store(C))

  if !combining
    if !isconj
      #println("!!! Doing uncombining post-processing step")
      rperm = ntuple(i -> (Nuc - i + 1), Nuc) # reverse permutation
      NDTensors.scale_blocks!(
        R, block -> NDTensors.permfactor(rperm, block, indsR; range=1:Nuc)
      )
    else
      #println("!!! Doing conjugate uncombining post-processing step")
      C_dir = dir(inds(C)[1])

      function mixed_arrow_sign(block)
        alpha_mixed_arrow = 1
        for n in 1:Nuc
          i = indsR[n]
          qi = qn(i, block[n])
          if dir(i) == C_dir && fparity(qi) == 1
            alpha_mixed_arrow *= -1
          end
        end
        return alpha_mixed_arrow
      end

      NDTensors.scale_blocks!(R, block -> mixed_arrow_sign(block))
    end
  end

  return R
end #NDTensors.after_combiner_signs
