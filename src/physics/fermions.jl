
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
  for i=1:L, j=i+1:L
    s *= sign(P[j]-P[i])
  end
  return s
end

isfermionic(qv::QNVal) = (modulus(qv) < 0)

isfermionic(qn::QN) = any(isfermionic,qn)
 
has_fermionic_subspaces(i::Index) = false

function has_fermionic_subspaces(i::QNIndex)
  for b=1:nblocks(i)
    isfermionic(qn(i,b)) && (return true)
  end
  return false
end

has_fermionic_subspaces(is::IndexSet) = false

function has_fermionic_subspaces(is::Union{QNIndexSet,NTuple{N,QNIndex}}) where {N}
  for i in is, b=1:nblocks(i)
    isfermionic(qn(i,b)) && (return true)
  end
  return false
end

has_fermionic_subspaces(T) = has_fermionic_subspaces(inds(T))

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
  return mod(p,2)
end

fparity(iv::IndexVal) = fparity(qn(iv))

Base.isodd(q::QN) = isodd(fparity(q))
Base.isodd(iv::IndexVal) = isodd(fparity(iv))

"""
    compute_permfactor(p,iv_or_qn::Vararg{T,N})

Given a permutation p and a set "s" of QNIndexVals or QNs,
if the subset of index vals which are fermion-parity
odd undergo an odd permutation (odd number of swaps)
according to p, then return -1. Otherwise return +1.
"""

function compute_permfactor(p,iv_or_qn::Vararg{T,N}) where {T,N}
  oddp = @MVector zeros(Int,N)
  n = 0
  for j=1:N
    if fparity(iv_or_qn[p[j]]) == 1
      n += 1
      oddp[n] = p[j]
    end
  end
  return parity_sign(oddp[1:n])
end

# Default implementation for non-QN IndexVals
NDTensors.permfactor(p,ivs...) where {N} = 1.0

NDTensors.permfactor(p,ivs::Vararg{QNIndexVal,N}) where {N} = compute_permfactor(p,ivs...)

function NDTensors.permfactor(p,pairs::Vararg{Pair{QNIndex,Int},N}) where {N} 
  ivs = ntuple(i->IndexVal(pairs[i]),N)
  return compute_permfactor(p,ivs...)
end

function NDTensors.permfactor(perm,
                              block::NDTensors.Block{N},
                              inds::Union{QNIndexSet,NTuple{N,QNIndex}}) where {N}
  qns = ntuple(n->qn(inds[n],block[n]),N)
  return compute_permfactor(perm,qns...)
end

#
# TODO: specialize this *just* for QNIndex as an optimization
#       (only remains to do this for input_indsR, if possible)
#       probably requires parameterizing IndexSet over the Index type
#
function NDTensors.compute_alpha(ElType,
                                 labelsR,blockR,input_indsR,
                                 labelsT1,blockT1,indsT1::NTuple{N1,QNIndex},
                                 labelsT2,blockT2,indsT2::NTuple{N2,QNIndex}) where {N1,N2}
  α = one(ElType)
  if !has_fermionic_subspaces(indsT1) || !has_fermionic_subspaces(indsT2)
    return α
  end

  # the "indsR" argument to compute_alpha from NDTensors
  # may be a tuple of QNIndex, so convert to an IndexSet
  indsR = IndexSet(input_indsR)
  
  orig_labelsT1 = [l for l in labelsT1]
  orig_labelsT2 = [l for l in labelsT2]

  nlabelsT1 = sort(orig_labelsT1;rev=true)
  nlabelsT2 = sort(orig_labelsT2)

  # Make orig_labelsR from the order of
  # indices that would result by just 
  # taking the uncontracted indices of
  # T1 and T2 in their input order:
  NR = length(labelsR)
  orig_labelsR = zeros(Int,NR)
  u = 1
  for l in (nlabelsT1...,nlabelsT2...)
    if l > 0 
      orig_labelsR[u] = l
      u += 1
    end
  end

  permT1 = NDTensors.getperm(nlabelsT1,orig_labelsT1)
  permT2 = NDTensors.getperm(nlabelsT2,orig_labelsT2)
  permR  = NDTensors.getperm(labelsR,orig_labelsR)

  alpha1 = NDTensors.permfactor(permT1,blockT1,indsT1)
  alpha2 = NDTensors.permfactor(permT2,blockT2,indsT2)
  alphaR = NDTensors.permfactor(permR,blockR,indsR)

  alpha_arrows = 1
  for n in 1:length(indsT1)
    l = orig_labelsT1[n]
    i = indsT1[n]
    qi = qn(i,blockT1[n])
    if l < 0 && dir(i)==Out && fparity(qi)==1
      alpha_arrows *= -1
    end
  end
  #@show alpha1,alpha2,alphaR,alpha_arrows

  α *= alpha1*alpha2*alphaR*alpha_arrows

  return α
end

# Flip signs of selected blocks of T prior to
# it being multiplied by a combiner ITensor
# labelsR gives the ordering of indices after the product
function NDTensors.mult_combiner_signs(C,
                                       labelsC_,
                                       indsC::NTuple{NC,QNIndex},
                                       T,
                                       labelsT_,
                                       indsT::NTuple{NT,QNIndex},
                                       labelsR_) where {NC,NT}
  #
  # Notes:
  #  - can use qn(i=>n) to get the QN of the subspace 
  #    corresponding to i=>n
  #  - could use convention of how combined ind is 
  #    mapped to uncombined to work out the "internal"
  #    parity of subspaces of the combined ind
  #    (similar to particle number mod 4)
  #  - can get setting of combined ind from blockT
  #    when looping over blockoffsets(T) or similar
  #    (or of uncombined ind when combining but then
  #     don't need to back out internal parity)
  #
  if !has_fermionic_subspaces(T)
    #println("Not copying T in combiner_signs")
    return T
  end


  #println("Fermionic case: copying T in combiner_signs")
  T = copy(T)

  labelsC = [l for l in labelsC_]
  labelsT = [l for l in labelsT_]

  # number of uncombined indices
  Nuc = NC-1

  ci = cinds(store(C))[1]
  combining = (labelsC[ci] > 0)

  isconj = NDTensors.isconj(store(C))

  if combining
    println("Combining <<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #(!isconj) ? println("Doing Case #1") : println("Doing Case #3")

    nlabelsT = Int[]

    if !isconj
      # Permute uncombined indices to front
      # in same order as indices passed to the
      # combiner constructor
      append!(nlabelsT,labelsC[2:end])
    else # isconj
      # If combiner is conjugated, put uncombined
      # indices in *opposite* order as on combiner
      append!(nlabelsT,reverse(labelsC[2:end]))
    end
    for l in labelsT
      if l > 0 #uncontracted
        append!(nlabelsT,l)
      end
    end

    @assert length(nlabelsT)==NT
    @show labelsC
    @show labelsT
    @show nlabelsT

    # Compute permutation that moves uncombined indices to front
    permT = NDTensors.getperm(nlabelsT,labelsT)
    @show permT

    for blockT in keys(blockoffsets(T))
      # Compute sign from permuting uncombined indices to front:
      alphaT = NDTensors.permfactor(permT,blockT,indsT)

      neg_dir = !isconj ? In : Out
      alpha_arrows = 1
      #alphaC = 1
      for n in 1:length(indsT)
        i = indsT[n]
        qi = qn(i,blockT[n])
        if labelsT[n] < 0 && fparity(qi)==1
          # vv DEBUG
          #arrow_sign = (dir(i)==neg_dir) ? -1 : +1
          #if arrow_sign == -1
          #  @show i,qi
          #end
          # ^^ DEBUG
          alpha_arrows *= (dir(i)==neg_dir) ? -1 : +1
          #alphaC *= -1
        end
      end

      fac = alphaT*alpha_arrows
      @show blockT,alphaT,alpha_arrows,fac

      #fac = alphaT
      #if !isconj
      #  fac *= alpha_arrows
      #  @show blockT,alphaT,alpha_arrows,fac
      #else
      #  @show blockT,alphaT,fac
      #end

      if fac != 1
        Tb = blockview(T,blockT)
        scale!(Tb,fac)
      end
    end # for blockT

  elseif !combining 
    #
    # Uncombining ---------------------------
    #
    #println("Uncombining >>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
    # Compute sign for permuting combined index to front
    # (sign alphaT to be computed for each block below):
    nlabelsT = sort(labelsT)
    permT = NDTensors.getperm(nlabelsT,labelsT)

    #
    # Note: other permutation of labelsT which
    # relates to two treatments of isconj==true/false
    # in combining case above is handled as a 
    # post-processing step in NDTensors src/blocksparse/combiner.jl
    #

    for blockT in keys(blockoffsets(T))
      alphaT = NDTensors.permfactor(permT,blockT,indsT)

      neg_dir = !isconj ? Out : In
      alpha_arrows = 1
      #alphaC = 1
      for n in 1:length(indsT)
        l = labelsT[n]
        i = indsT[n]
        if l < 0 && fparity(qn(i,blockT[n]))==1
          alpha_arrows = (dir(i)==neg_dir) ? -1 : +1
          #alphaC = -1
          break 
        end
      end

      #@show blockT,alphaT,alpha_arrows
      fac = alphaT#*alpha_arrows
      if fac != 1
        Tb = blockview(T,blockT)
        scale!(Tb,fac)
      end
    end
  end

  return T
end #NDTensors.mult_combiner_signs
