export polar,
       eigenHermitian,
       factorize

import LinearAlgebra.qr
function qr(A::ITensor,
            Linds...;
            kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,qr")
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  QT,RT = qr(Tensor(A),Lpos,Rpos)
  Q,R = ITensor(QT),ITensor(RT)
  q = commonindex(Q,R)
  settags!(Q,tags,q)
  settags!(R,tags,q)
  q = settags(q,tags)
  return Q,R,q
end

# TODO: allow custom tags in internal indices?
function polar(A::ITensor,
               Linds...;
               kwargs...)
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,PT = polar(Tensor(A),Lpos,Rpos)
  U,P = ITensor(UT),ITensor(PT)
  u = commoninds(U,P)
  p = uniqueinds(P,U)
  replaceinds!(U,u,p')
  replaceinds!(P,u,p')
  return U,P,commoninds(U,P)
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
* `maxdim` [Int]
* `mindim` [Int]
* `cutoff` [Float64]
* `absoluteCutoff` [Bool] Default value: false.
* `doRelCutoff` [Bool] Default value: true.
* `utags` [String] Default value: "Link,u".
* `vtags` [String] Default value: "Link,v".
* `fastSVD` [Bool] Defaut value: false.
"""
function svd(A::ITensor,
             Linds...;
             kwargs...)
  utags::TagSet = get(kwargs,:utags,"Link,u")
  vtags::TagSet = get(kwargs,:vtags,"Link,v")
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,ST,VT = svd(Tensor(A),Lpos,Rpos;kwargs...)
  U,S,V = ITensor(UT),ITensor(ST),ITensor(VT)
  u = commonindex(U,S)
  v = commonindex(S,V)
  settags!(U,utags,u)
  settags!(S,utags,u)
  settags!(S,vtags,v)
  settags!(V,vtags,v)
  u = settags(u,utags)
  v = settags(v,vtags)
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
  Lis = commoninds(inds(A),IndexSet(Linds...))
  A² = A*prime(dag(A),Lis)
  FU,D = eigenHermitian(A²,Lis,prime(Lis);kwargs...)
  FV = dag(FU)*A
  return FU,FV,commonindex(FU,FV)
end

function _factorize_from_right_eigen(A::ITensor,
                                     Linds...; 
                                     kwargs...)
  Ris = uniqueinds(inds(A),IndexSet(Linds...))
  A² = A*prime(dag(A),Ris)
  FV,D = eigenHermitian(A²,Ris,prime(Ris); kwargs...)
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
  throw(ArgumentError("In factorize, no dir = $dir supported. Use center, fromleft or fromright."))
end

function eigenHermitian(A::ITensor,
                        Linds=findinds(A,"0"),
                        Rinds=prime(IndexSet(Linds));
                        kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,eigen")
  lefttags::TagSet = get(kwargs,:lefttags,tags)
  righttags::TagSet = get(kwargs,:righttags,prime(tags))
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,DT = eigenHermitian(Tensor(A),Lpos,Rpos;kwargs...)
  U,D = ITensor(UT),ITensor(DT)
  u = commonindex(U,D)
  settags!(U,lefttags,u)
  settags!(D,lefttags,u)
  u = settags(u,lefttags)
  v = uniqueindex(D,U)
  D *= δ(v,settags(u,righttags))
  return U,D,u,v
end

import LinearAlgebra.eigen
function eigen(A::ITensor,
               Linds=findinds(A,"0"),
               Rinds=prime(IndexSet(Linds));
               kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,eigen")
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,DT = eigen(Tensor(A),Lpos,Rpos;kwargs...)
  U,D = ITensor(UT),ITensor(DT)
  u = commonindex(U,D)
  settags!(U,tags,u)
  settags!(D,tags,u)
  u = settags(u,tags)
  v = uniqueindex(D,U)
  D *= δ(v,u')
  return U,D,u
end

