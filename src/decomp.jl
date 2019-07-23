export svd,
       qr,
       polar,
       eigen,
       factorize

function truncate!(P::Vector{Float64};
                   kwargs...)::Tuple{Float64,Float64}
  maxdim::Int = get(kwargs,:maxdim,length(P))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)

  origm = length(P)
  docut = 0.0

  if P[1]==0.0
    resize!(P,1)
    return 0.,0.
  end

  if origm==1
    docut = P[1]/2
    return 0.,docut
  end

  #Zero out any negative weight
  for n=origm:-1:1
    (P[n] >= 0.0) && break
    P[n] = 0.0
  end
  
  n = origm
  truncerr = 0.0
  while n > maxdim
    truncerr += P[n]
    n -= 1
  end

  if absoluteCutoff
    #Test if individual prob. weights fall below cutoff
    #rather than using *sum* of discarded weights
    while P[n] <= cutoff && n > mindim
      truncerr += P[n]
      n -= 1
    end
  else
    scale = 1.0
    if doRelCutoff
      scale = sum(P)
      (scale==0.0) && (scale = 1.0)
    end

    #@show P, truncerr
    #@show length(P)
    #Continue truncating until *sum* of discarded probability 
    #weight reaches cutoff reached (or m==mindim)
    while (truncerr+P[n] <= cutoff*scale) && (n > mindim)
      truncerr += P[n]
      n -= 1
    end
    #@show n, truncerr 
    #println()
    #println()
    if scale==0.0
      truncerr = 0.0
    else
      truncerr /= scale
    end
  end

  if n < 1
    n = 1
  end
  if n < origm
    docut = (P[n]+P[n+1])/2
    if abs(P[n]-P[n+1]) < 1E-3*P[n]
      docut += 1E-3*P[n]
    end
  end

  resize!(P,n)

  return truncerr,docut
end

function _permute_for_factorize(A::ITensor,
                                Linds...)
  Ais = inds(A)
  Lis_orig = IndexSet(Linds...)
  Lis = commoninds(Ais,Lis_orig)
  Ris = uniqueinds(Ais,Lis)
  Ais_perm = IndexSet(Lis...,Ris...)
  # TODO: check if hassameinds(Lis,Ais[1:length(Lis)])
  # so that a permute can be avoided
  if inds(A) ≠ Ais_perm
    A = permute(A,Ais_perm)
  end
  return A,Lis,Ris
end

import LinearAlgebra.qr
function qr(A::ITensor,
            Linds...)
  A,Lis,Ris = _permute_for_factorize(A,Linds...)
  Qis,Qstore,Pis,Pstore = storage_qr(store(A),Lis,Ris)
  return ITensor(Qis,Qstore),ITensor(Pis,Pstore)
end

function polar(A::ITensor,
               Linds...)
  A,Lis,Ris = _permute_for_factorize(A,Linds...)
  Qis,Qstore,Pis,Pstore = storage_polar(store(A),Lis,Ris)
  return ITensor(Qis,Qstore),ITensor(Pis,Pstore)
end

import LinearAlgebra.svd
"""
    svd(A::ITensor,
        leftind1::Index,
        leftind2::Index,
        ...
        ;kwargs...)

Singular value decomposition (SVD) of an ITensor A, computed
by treating the "left indices" provided collectively
as a row index, and the remaining "right indices" as a
column index (matricization of a tensor).

Whether the SVD performs a trunction depends on the keyword
arguments provided. The following keyword arguments are recognized:
* maxdim [Int]
* mindim [Int]
* cutoff [Float64]
* truncate [Bool]
"""
function svd(A::ITensor,
             Linds...;
             kwargs...)
  A,Lis,Ris = _permute_for_factorize(A,Linds...)
  Uis,Ustore,Sis,Sstore,Vis,Vstore = storage_svd(store(A),Lis,Ris;kwargs...)

  U = ITensor(Uis,Ustore)
  S = ITensor(Sis,Sstore)
  V = ITensor(Vis,Vstore)
  u = commonindex(U,S)
  v = commonindex(S,V)
  return U,S,V,u,v
end

function _factorize_center(A::ITensor,
                           Linds...;
                           kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,u")
  U,S,V = svd(A,Linds...;kwargs...)
  u = commonindex(U,S)
  v = commonindex(S,V)
  for ss = 1:dim(u)
    S[ss,ss] = sqrt(S[ss,ss])
  end
  FU = settags(U*S,tags,v)
  FV = settags(S*V,tags,u)
  return FU,FV,commonindex(FU,FV)
end

function _factorize_from_left_svd(A::ITensor,
                                  Linds...;
                                  kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,u")
  U,S,V = svd(A,Linds...;kwargs...)
  u = commonindex(U,S)
  FU = settags(U,tags,u)
  FV = settags(S*V,tags,u)
  return FU,FV,commonindex(FU,FV)
end

function _factorize_from_right_svd(A::ITensor,
                                   Linds...; 
                                   kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,u")
  U,S,V = svd(A,Linds...;kwargs...)
  v = commonindex(S,V)
  FU = settags(U*S,tags,v)
  FV = settags(V,tags,v)
  return FU,FV,commonindex(FU,FV)
end

function _factorize_from_left_eigen(A::ITensor,
                                    Linds...; 
                                    kwargs...)
  A,Lis,Ris = _permute_for_factorize(A,Linds...)
  A² = A*prime(dag(A),Lis)
  FU,D = eigen(A²,Lis,prime(Lis);kwargs...)
  FV = dag(FU)*A
  return FU,FV,commonindex(FU,FV)
end

function _factorize_from_right_eigen(A::ITensor,
                                     Linds...; 
                                     kwargs...)
  A,Lis,Ris = _permute_for_factorize(A,Linds...)
  A² = A*prime(dag(A),Ris)
  FV,D = eigen(A²,Ris,prime(Ris); kwargs...)
  FU = A*dag(FV)
  return FU,FV,commonindex(FU,FV)
end

import LinearAlgebra.factorize
"""
factorize(A::ITensor, Linds...; dir = "center", which_factorization = "svd", cutoff = 0.0)

Do a low rank factorization of A either using an SVD or an eigendecomposition of A'A or AA'.
"""
function factorize(A::ITensor,
                   Linds...;
                   kwargs...)
  dir::String = get(kwargs,:dir,"center")
  if dir == "center"
    return _factorize_center(A,Linds...;kwargs...)
  end
  which_factorization::String = get(kwargs,:which_factorization,"svd")
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  use_eigen = false
  if which_factorization == "eigen" || (which_factorization == "automatic" && cutoff > 1e-12)
    use_eigen = true
  end
  if dir == "fromleft"
    if use_eigen
      return _factorize_from_left_eigen(A,Linds...;kwargs...)
    else
      return _factorize_from_left_svd(A,Linds...;kwargs...)
    end
  elseif dir == "fromright"
    if use_eigen
      return _factorize_from_right_eigen(A,Linds...;kwargs...)
    else
      return _factorize_from_right_svd(A,Linds...;kwargs...)
    end
  end
  error("In factorize, no dir = $dir supported. Use center, fromleft or fromright.")
end

# TODO: add a version that automatically detects the IndexSets
# by matching based on tags
import LinearAlgebra.eigen
function eigen(A::ITensor,
               Linds,
               Rinds;
               kwargs...)
  Lis = IndexSet(Linds)
  Ris = IndexSet(Rinds)
  Ais_perm = IndexSet(Lis...,Ris...)
  !hassameinds(Ais_perm,A) && throw(ErrorException("Input indices must be contained in the ITensor"))
  if inds(A) ≠ Ais_perm
    A = permute(A,Ais_perm)
  end
  #TODO: More of the index analysis should be moved out of storage_eigen
  Uis,Ustore,Dis,Dstore = storage_eigen(store(A),Lis,Ris; kwargs...)
  return ITensor(Uis,Ustore),ITensor(Dis,Dstore)
end

