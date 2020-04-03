export factorize,
       eigen,
       qr,
       svd

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
  Lis = commoninds(A,IndexSet(Linds...))
  Ris = uniqueinds(A,Lis)

  CL,cL = combiner(Lis...)
  CR,cR = combiner(Ris...)

  AC = A*CR*CL

  if inds(AC) != IndexSet(cL,cR)
    AC = permute(AC,cL,cR)
  end

  UT,ST,VT,spec = svd(tensor(AC);kwargs...)
  UC,S,VC = itensor(UT),itensor(ST),itensor(VT)

  u = commonind(S,UC)
  v = commonind(S,VC)

  if hasqns(A)
    # Fix the flux of UC,S,VC
    # such that flux(UC) == flux(VC) == QN()
    # and flux(S) == flux(A)
    for b in nzblocks(UC)
      i1 = inds(UC)[1]
      i2 = inds(UC)[2]
      newqn = -dir(i2)*qn(i1,b[1])
      setblockqn!(i2,newqn,b[2])
      setblockqn!(u,newqn,b[2])
    end

    for b in nzblocks(VC)
      i1 = inds(VC)[1]
      i2 = inds(VC)[2]
      newqn = -dir(i2)*qn(i1,b[1])
      setblockqn!(i2,newqn,b[2])
      setblockqn!(v,newqn,b[2])
    end
  end

  U = UC*dag(CL)
  V = VC*dag(CR)

  settags!(U,utags,u)
  settags!(S,utags,u)
  settags!(S,vtags,v)
  settags!(V,vtags,v)

  u = settags(u,utags)
  v = settags(v,vtags)

  return TruncSVD(U,S,V,spec,u,v)
end

"""
  TruncEigen{N}
ITensor factorization type for a truncated eigenvalue decomposition, returned by
`eigen`.
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

function LinearAlgebra.eigen(A::ITensor,
                             Linds=inds(A;plev=0),
                             Rinds=prime(IndexSet(Linds));
                             kwargs...)
  ishermitian::Bool = get(kwargs,:ishermitian,false)
  tags::TagSet = get(kwargs,:tags,"Link,eigen")
  lefttags::TagSet = get(kwargs,:lefttags,tags)
  righttags::TagSet = get(kwargs,:righttags,tags)
  leftplev = get(kwargs,:leftplev,0)
  rightplev = get(kwargs,:rightplev,lefttags==righttags ? 1 : 0)

  (lefttags==righttags && leftplev==rightplev) && error("In eigen, left tags and prime level must be different from right tags and prime level")

  Lis = commoninds(A,IndexSet(Linds))
  Ris = commoninds(A,IndexSet(Rinds))

  CL,cL = combiner(Lis...)
  CR,cR = combiner(Ris...)

  AC = A*CR*CL

  if inds(AC) != IndexSet(cL,cR)
    AC = permute(AC,cL,cR)
  end

  AT = ishermitian ? Hermitian(tensor(AC)) : tensor(AC)
  UT,DT,spec = eigen(AT;kwargs...)
  UC,D = itensor(UT),itensor(DT)

  u = commonind(UC,D)

  if hasqns(A)
    # Fix the flux of UC,D
    # such that flux(UC) == QN()
    # and flux(D) == flux(A)
    for b in nzblocks(UC)
      i1 = inds(UC)[1]
      i2 = inds(UC)[2]
      newqn = -dir(i2)*qn(i1,b[1])
      setblockqn!(i2,newqn,b[2])
      setblockqn!(u,newqn,b[2])
    end
  end

  U = UC*dag(CL)

  # Set left index tags
  u = commonind(D,U)
  settags!(U,lefttags,u)
  settags!(D,lefttags,u)

  # Set left index plev
  u = commonind(D,U)
  U = setprime(U,leftplev,u)
  D = setprime(D,leftplev,u)

  # Set right index tags and plev
  v = uniqueind(D,U)
  replaceind!(D,v,setprime(settags(u,righttags),rightplev))

  u = commonind(D,U) 
  v = uniqueind(D,U)
  return TruncEigen(U,D,spec,u,v)
end

function LinearAlgebra.qr(A::ITensor,
                          Linds...;
                          kwargs...)
  tags::TagSet = get(kwargs,:tags,"Link,qr")
  Lis = commoninds(A,IndexSet(Linds...))
  Ris = uniqueinds(A,Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  QT,RT = qr(tensor(A),Lpos,Rpos;kwargs...)
  Q,R = itensor(QT),itensor(RT)
  q = commonind(Q,R)
  settags!(Q,tags,q)
  settags!(R,tags,q)
  q = settags(q,tags)
  return Q,R,q
end

# TODO: allow custom tags in internal indices?
function Tensors.polar(A::ITensor,
                       Linds...;
                       kwargs...)
  Lis = commoninds(A,IndexSet(Linds...))
  Ris = uniqueinds(A,Lis)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  UT,PT = polar(tensor(A),Lpos,Rpos)
  U,P = itensor(UT),itensor(PT)
  u = commoninds(U,P)
  p = uniqueinds(P,U)
  replaceinds!(U,u,p')
  replaceinds!(P,u,p')
  return U,P,commoninds(U,P)
end

function factorize_svd(A::ITensor,
                       Linds...;
                       kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  tags::TagSet = get(kwargs, :tags, "Link,fact")
  U,S,V,spec,u,v = svd(A, Linds...; kwargs...)
  if ortho == "left"
    L,R = U,S*V
  elseif ortho == "right"
    L,R = U*S,V
  elseif ortho == "none"
    sqrtS = S
    sqrtS .= sqrt.(S)
    L,R = U*sqrtS,sqrtS*V
    replaceind!(L,v,u)
  else
    error("In factorize using svd decomposition, ortho keyword $ortho not supported. Supported options are left, right, or none.")
  end

  # Set the tags properly
  l = commonind(L,R)
  settags!(L, tags, l)
  settags!(R, tags, l)
  l = settags(l, tags)

  return L,R,spec,l
end

function factorize_eigen(A::ITensor,
                         Linds...;
                         kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  if ortho == "left"
    Lis = commoninds(A,IndexSet(Linds...))
    A² = A*prime(dag(A),Lis)
    L,D,spec = eigen(A²,Lis,prime(Lis); ishermitian=true,
                                        kwargs...)
    R = dag(L)*A
  elseif ortho == "right"
    Ris = uniqueinds(A,IndexSet(Linds...))
    A² = A*prime(dag(A),Ris)
    R,D,spec = eigen(A²,Ris,prime(Ris); ishermitian=true,
                                        kwargs...)
    L = A*dag(R)
  else
    error("In factorize using eigen decomposition, ortho keyword $ortho not supported. Supported options are left or right.")
  end
  return L,R,spec,commonind(L,R)
end

"""
factorize(A::ITensor, Linds...;
          ortho = "left",
          which_decomp = "automatic",
          tags = "Link,fact",
          cutoff = 0.0,
          maxdim = ...)

Perform a factorization of A into ITensors L and R such the A ≈ L*R.

Choose orthogonality properties of the factorization with the keyword `ortho`. For example, if `ortho = "left"`, the left factor L is an orthogonal basis such that `L * dag(prime(L, commonind(L,R))) ≈ I`. If `ortho = "right"`, the right factor R forms an orthogonal basis. Finally, if `ortho = "none"`, neither of the factors form an orthogonal basis, and in general are made as symmetricly as possible (based on the decomposition used).

By default, the decomposition used is chosen automatically. You can choose which decomposition to use with the keyword `which_decomp`. Right now, options `"svd"` and `"eigen"` are supported.

When `"svd"` is chosen, L = U and R = S*V for `ortho = "left"`, L = U*S and R = V for `ortho = "right"`, and L = U*sqrt(S) and R = sqrt(S)*V for `ortho = "none"`.

When `"eigen"` is chosen, L = U and R = U'*A where U is determined
from the eigendecompositon A*A' = U*D*U' for `ortho = "left"` (and vice versa for `ortho = "right"`). `"eigen"` is not supported for `ortho = "none"`. 

When `"automatic"` is chosen, svd or eigen is used depending on the provided cutoff (eigen is only used when the cutoff is greater than 1e-12, since it has a lower precision).

In the future, other decompositions like QR, polar, cholesky, LU, etc.
are expected to be supported.
"""
function LinearAlgebra.factorize(A::ITensor,
                                 Linds...;
                                 kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  which_decomp::String = get(kwargs, :which_decomp, "automatic")
  cutoff::Float64 = get(kwargs, :cutoff, 0.0)

  # Deprecated keywords
  haskey(kwargs, :dir) && 
  error("""dir keyword in factorize has been replace by ortho.
  Note that the default is now `left`, meaning for the results L,R = factorize(A), L forms an orthogonal basis.""")

  haskey(kwargs, :which_factorization) && 
  error("""which_factorization keyword in factorize has been replace by which_decomp.""")

  # Determines when to use eigen vs. svd (eigen is less precise,
  # so eigen should only be used if a larger cutoff is requested)
  automatic_cutoff = 1e-12
  if which_decomp == "svd" || 
     (which_decomp == "automatic" && cutoff ≤ automatic_cutoff)
    L,R,spec,l = factorize_svd(A, Linds...; kwargs...)
  elseif which_decomp == "eigen" ||
         (which_decomp == "automatic" && cutoff > automatic_cutoff)
    L,R,spec,l = factorize_eigen(A, Linds...; kwargs...)
  else
    return throw(ArgumentError("In factorize, no factorization $which_decomp supported. Use svd, eigen, or automatic."))
  end
  return L,R,spec,l
end

