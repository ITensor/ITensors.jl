export factorize,
       eigen,
       qr,
       svd

function LinearAlgebra.qr(A::ITensor,
                          Linds...;
                          kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,qr")
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  QT,RT = qr(tensor(A),Lpos,Rpos;kwargs...)
  Q,R = ITensor(QT),ITensor(RT)
  q = commonindex(Q,R)
  settags!(Q,tags,q)
  settags!(R,tags,q)
  q = settags(q,tags)
  return Q,R,q
end

# TODO: allow custom tags in internal indices?
function Tensors.polar(A::ITensor,
                       Linds...;
                       kwargs...)
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,PT = polar(tensor(A),Lpos,Rpos)
  U,P = ITensor(UT),ITensor(PT)
  u = commoninds(U,P)
  p = uniqueinds(P,U)
  replaceinds!(U,u,p')
  replaceinds!(P,u,p')
  return U,P,commoninds(U,P)
end
"""
  TruncSVD{N}
ITensor factorization type for a truncated singular-value decomposition, returned by
`svd`.
"""
struct TruncSVD{N1,N2}
  U::ITensor{N1}
  S::ITensor{2}
  V::ITensor{N2}
  spec::Spectrum
  u::Index
  v::Index
end

# iteration for destructuring into components `U,S,V,spec,u,v = S`
Base.iterate(S::TruncSVD) = (S.U, Val(:S))
Base.iterate(S::TruncSVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::TruncSVD, ::Val{:V}) = (S.V, Val(:spec))
Base.iterate(S::TruncSVD, ::Val{:spec}) = (S.spec, Val(:u))
Base.iterate(S::TruncSVD, ::Val{:u}) = (S.u, Val(:v))
Base.iterate(S::TruncSVD, ::Val{:v}) = (S.v, Val(:done))
Base.iterate(S::TruncSVD, ::Val{:done}) = nothing

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
function LinearAlgebra.svd(A::ITensor,
                           Linds...;
                           kwargs...)
  utags::TagSet = get(kwargs,:utags,"Link,u")
  vtags::TagSet = get(kwargs,:vtags,"Link,v")
  Lis = commoninds(inds(A),IndexSet(Linds...))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,ST,VT,spec = svd(tensor(A),Lpos,Rpos;kwargs...)
  U,S,V = ITensor(UT),ITensor(ST),ITensor(VT)
  u₀ = commonindex(U,S)
  v₀ = commonindex(S,V)

  u = settags(u₀,utags)
  v = settags(u₀,vtags)

  U *= δ(dag(u₀),u)
  S = δ(dag(u₀),u)*S*δ(dag(v₀),v)
  V *= δ(dag(v₀),v)

  return TruncSVD(U,S,V,spec,u,v)
end

function _factorize_center(A::ITensor,
                           Linds...;
                           kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,u")
  U,S,V,spec = svd(A,Linds...;kwargs...)
  u = commonindex(U,S)
  v = commonindex(S,V)
  for ss = 1:dim(u)
    S[ss,ss] = sqrt(S[ss,ss])
  end
  FU = settags(U*S,tags,v)
  FV = settags(S*V,tags,u)
  return FU,FV,spec,commonindex(FU,FV)
end

function _factorize_from_left_svd(A::ITensor,
                                  Linds...;
                                  kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,u")
  U,S,V,spec = svd(A,Linds...;kwargs...)
  u = commonindex(U,S)
  FU = settags(U,tags,u)
  FV = settags(S*V,tags,u)
  return FU,FV,spec,commonindex(FU,FV)
end

function _factorize_from_right_svd(A::ITensor,
                                   Linds...;
                                   kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,u")
  U,S,V,spec = svd(A,Linds...;kwargs...)
  v = commonindex(S,V)
  FU = settags(U*S,tags,v)
  FV = settags(V,tags,v)
  return FU,FV,spec,commonindex(FU,FV)
end

function _factorize_from_left_eigen(A::ITensor,
                                    Linds...;
                                    kwargs...)
  Lis = commoninds(inds(A),IndexSet(Linds...))
  A² = A*prime(dag(A),Lis)
  FU,D,spec = eigenHermitian(A²,Lis,prime(Lis); ispossemidef=true,
                                           kwargs...)
  FV = dag(FU)*A
  return FU,FV,spec,commonindex(FU,FV)
end

function _factorize_from_right_eigen(A::ITensor,
                                     Linds...;
                                     kwargs...)
  Ris = uniqueinds(inds(A),IndexSet(Linds...))
  A² = A*prime(dag(A),Ris)
  FV,D,spec = eigenHermitian(A²,Ris,prime(Ris); ispossemidef=true,
                             kwargs...)
  FU = A*dag(FV)
  return FU,FV,spec,commonindex(FU,FV)
end

"""
factorize(A::ITensor, Linds...; dir = "center", which_factorization = "svd", cutoff = 0.0)

Do a low rank factorization of A either using an SVD or an eigendecomposition of A'A or AA'.
"""
function LinearAlgebra.factorize(A::ITensor,
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

"""
  TruncEigen{N}
ITensor factorization type for a truncated eigenvalue decomposition, returned by
`eigenHermitian`.
"""
struct TruncEigen{N}
  U::ITensor{N}
  D::ITensor{2}
  spec::Spectrum
  u::Index
  v::Index
end

# iteration for destructuring into components `U,D,spec,u,v = E`
Base.iterate(E::TruncEigen) = (E.U, Val(:D))
Base.iterate(E::TruncEigen, ::Val{:D}) = (E.D, Val(:spec))
Base.iterate(E::TruncEigen, ::Val{:spec}) = (E.spec, Val(:u))
Base.iterate(E::TruncEigen, ::Val{:u}) = (E.u, Val(:v))
Base.iterate(E::TruncEigen, ::Val{:v}) = (E.v, Val(:done))
Base.iterate(E::TruncEigen, ::Val{:done}) = nothing

function Tensors.eigenHermitian(A::ITensor,
                                Linds=findinds(A,("",0)),
                                Rinds=prime(IndexSet(Linds));
                                kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,eigen")
  lefttags::TagSet = get(kwargs,:lefttags,tags)
  righttags::TagSet = get(kwargs,:righttags,prime(tags))
  Lis = commoninds(inds(A),IndexSet(Linds))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,DT,spec = eigenHermitian(tensor(A),Lpos,Rpos;kwargs...)
  U,D = ITensor(UT),ITensor(DT)
  u = commonindex(U,D)
  settags!(U,lefttags,u)
  settags!(D,lefttags,u)
  u = settags(u,lefttags)
  v = uniqueindex(D,U)
  D *= δ(v,settags(u,righttags))
  return TruncEigen(U,D,spec,u,v)
end

function LinearAlgebra.eigen(A::ITensor,
                             Linds=findinds(A,("",0)),
                             Rinds=prime(IndexSet(Linds));
                             kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,eigen")
  Lis = commoninds(inds(A),IndexSet(Linds))
  Ris = uniqueinds(inds(A),Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,DT = eigen(tensor(A),Lpos,Rpos;kwargs...)
  U,D = ITensor(UT),ITensor(DT)
  u = commonindex(U,D)
  settags!(U,tags,u)
  settags!(D,tags,u)
  u = settags(u,tags)
  v = uniqueindex(D,U)
  D *= δ(v,u')
  return U,D,u
end

