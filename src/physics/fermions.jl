
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

function has_fermionic_subspaces(is::QNIndexSet)
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

internal_factor(block::NTuple{N,Int},inds) where {N} = 1

function internal_factor(block::NTuple{N,Int},inds::QNIndexSet) where {N}
  qns = ntuple(n->qn(inds[n],block[n]),N)
  fac = 1
  for q in qns
    for v in q
      !isactive(v) && break
      if isfermionic(v) && mod(abs(val(v)),4) >= 2
        fac *= -1
      end
      #@show q,v,fac
    end
  end
  return fac
end

#
# TODO: specialize this *just* for QNIndex as an optimization
#       (only remains to do this for input_indsR, if possible)
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
  #@show alpha1,alpha2,alphaR,alpha_arrows

  return alpha1*alpha2*alphaR*alpha_arrows
end

# Flip signs of selected blocks of T prior to
# it being multiplied by a combiner ITensor
# labelsR gives the ordering of indices after the product
function NDTensors.mult_combiner_signs(C,
                                       labelsC,indsC::QNIndexSet,
                                       T,
                                       labelsT,indsT::QNIndexSet,
                                       labelsR)
  if !has_fermionic_subspaces(T)
    println("Not copying T in combiner_signs")
    return T
  end

  println("Fermionic case: copying T in combiner_signs")
  T = copy(T)

  orig_labelsC = [l for l in labelsC]
  orig_labelsT = [l for l in labelsT]

  #NR = length(labelsR)
  #orig_labelsR = zeros(Int,NR)
  #u = 1
  #for l in (nlabelsC...,nlabelsT...)
  #  if l > 0 
  #    orig_labelsR[u] = l
  #    u += 1
  #  end
  #end

  ci = cinds(store(C))[1]
  combining = (orig_labelsC[ci] > 0)
  @show combining

  isconj = NDTensors.isconj(store(C))
  @show isconj

  #@show orig_labelsC
  #@show orig_labelsT
  #@show orig_labelsR
  #@show nlabelsC
  #@show nlabelsT
  #@show labelsR

  #permC = NDTensors.getperm(tuple(nlabelsC...),tuple(orig_labelsC...))
  #permT = NDTensors.getperm(tuple(nlabelsT...),tuple(orig_labelsT...))
  #permR = NDTensors.getperm(tuple(labelsR...),tuple(orig_labelsR...))

  # NOTES:
  # X already included alphaT below
  # - handle alphaC by just assuming combiner is either:
  #    > not permuted at all for regular case
  #    > reverse-permuted for dagger case (how to determine combiner block then?)
  # - use assumptions about combiner logic to simplify alphaR?
  # INITIAL DESIGN:
  # - for un-daggered combiner, assume:
  #   > permC is trivial i.e. alphaC is +1 always
  #   > permR is also trivial so alphaR is +1
  #   > thus signs come only from alphaT and arrows
  # - for daggered case: ...

  # <Case #1>
  if combining && !isconj

    nlabelsT = sort(orig_labelsT)
    permT = NDTensors.getperm(tuple(nlabelsT...),tuple(orig_labelsT...))

    for (blockT,_) in blockoffsets(T)
      alphaT = NDTensors.permfactor(permT,blockT,indsT)

      alpha_arrows = 1
      for n in 1:length(indsT)
        l = orig_labelsT[n]
        i = indsT[n]
        qi = qn(i,blockT[n])
        if l < 0 && dir(i)==In && fparity(qi)==1
          alpha_arrows *= -1
        end
      end
      fac = alphaT*alpha_arrows
      if fac != 1
        Tb = blockview(T,blockT)
        scale!(Tb,fac)
      end
    end
  # <Case #2>:
  elseif !combining && !isconj
    nlabelsT = sort(orig_labelsT;rev=true)
    nlabelsC = sort(orig_labelsC)
    @show labelsC
    @show labelsT
    @show nlabelsT
    @show labelsR

    NR = length(labelsR)
    orig_labelsR = zeros(Int,NR)
    u = 1
    for l in (nlabelsT...,nlabelsC...)
      if l > 0 
        orig_labelsR[u] = l
        u += 1
      end
    end
    @show orig_labelsR

    permT = NDTensors.getperm(tuple(nlabelsT...),tuple(orig_labelsT...))
    permR = NDTensors.getperm(tuple(labelsR...),tuple(orig_labelsR...))
    @show permR

    for (blockT,_) in blockoffsets(T)
      @show permT
      @show blockT
      @show indsT
      alphaT = NDTensors.permfactor(permT,blockT,indsT)

      #
      # TODO:
      # Issue here is whether "indsR" below and blockR
      # can be built from information of combined version of T
      # or whether we need to loop over blocks of *uncombined*
      # version of T - maybe the second, but still get alphaT
      # from the labels use prior to uncombining
      #
      #alphaR = NDTensors.permfactor(permR,blockR,indsR)

      alpha_arrows = 1
      for n in 1:length(indsT)
        l = orig_labelsT[n]
        i = indsT[n]
        qi = qn(i,blockT[n])
        #TODO: check that Out here is the correct direction:
        if l < 0 && dir(i)==Out && fparity(qi)==1
          alpha_arrows *= -1
        end
      end

      #TODO: include alphaR here:
      fac = alphaT*alpha_arrows#*alphaR
      @show fac
      if fac != 1
        Tb = blockview(T,blockT)
        scale!(Tb,fac)
      end
    end
  end

  return T
end
