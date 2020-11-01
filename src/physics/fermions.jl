
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
 
has_fermionic_sectors(i::Index) = false

function has_fermionic_sectors(i::QNIndex)
  for b=1:nblocks(i)
    isfermionic(qn(i,b)) && (return true)
  end
  return false
end

has_fermionic_sectors(is::IndexSet) = false

function has_fermionic_sectors(is::QNIndexSet)
  for i in is, b=1:nblocks(i)
    isfermionic(qn(i,b)) && (return true)
  end
  return false
end

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

function NDTensors.permfactor(p,block::NTuple{N,Int},inds::QNIndexSet) where {N}
  qns = ntuple(n->qn(inds[n],block[n]),N)
  return compute_permfactor(p,qns...)
end

#
# TODO: specialize this *just* for QNIndex as an optimization
#       probably requires parameterizing IndexSet over the Index type
#
function NDTensors.compute_alpha(labelsR,blockR,input_indsR,
                                 labelsT1,blockT1,indsT1::QNIndexSet,
                                 labelsT2,blockT2,indsT2::QNIndexSet)
  # the "indsR" argument to compute_alpha from NDTensors
  # may be a tuple of QNIndex, so convert to an IndexSet
  indsR = IndexSet(input_indsR)
  
  orig_labelsT1 = [l for l in labelsT1]
  orig_labelsT2 = [l for l in labelsT2]

  nlabelsT1 = sort(orig_labelsT1;rev=true)
  nlabelsT2 = sort(orig_labelsT2)

  NR = length(labelsR)
  orig_labelsR = zeros(Int,NR)
  u = 1
  for l in (nlabelsT1...,nlabelsT2...)
    if l > 0 
      orig_labelsR[u] = l
      u += 1
    end
  end

  permT1 = NDTensors.getperm(tuple(nlabelsT1...),tuple(orig_labelsT1...))
  permT2 = NDTensors.getperm(tuple(nlabelsT2...),tuple(orig_labelsT2...))
  permR  = NDTensors.getperm(tuple(labelsR...),tuple(orig_labelsR...))

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

  return alpha1*alpha2*alphaR*alpha_arrows
end

