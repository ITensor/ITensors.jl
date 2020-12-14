
"""
    TruncSVD{N}

ITensor factorization type for a truncated singular-value 
decomposition, returned by `svd`.
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
iterate(S::TruncSVD) = (S.U, Val(:S))
iterate(S::TruncSVD, ::Val{:S}) = (S.S, Val(:V))
iterate(S::TruncSVD, ::Val{:V}) = (S.V, Val(:spec))
iterate(S::TruncSVD, ::Val{:spec}) = (S.spec, Val(:u))
iterate(S::TruncSVD, ::Val{:u}) = (S.u, Val(:v))
iterate(S::TruncSVD, ::Val{:v}) = (S.v, Val(:done))
iterate(S::TruncSVD, ::Val{:done}) = nothing

@doc """
    svd(A::ITensor, inds::Index...; <keyword arguments>)

Singular value decomposition (SVD) of an ITensor `A`, computed
by treating the "left indices" provided collectively
as a row index, and the remaining "right indices" as a
column index (matricization of a tensor).

The first three return arguments are `U`, `S`, and `V`, such that
`A ≈ U * S * V`.

Whether or not the SVD performs a trunction depends on the keyword
arguments provided. 

# Arguments
- `maxdim::Int`: the maximum number of singular values to keep.
- `mindim::Int`: the minimum number of singular values to keep.
- `cutoff::Float64`: set the desired truncation error of the SVD, by default defined as the sum of the squares of the smallest singular values.
- `lefttags::String = "Link,u"`: set the tags of the Index shared by `U` and `S`.
- `righttags::String = "Link,v"`: set the tags of the Index shared by `S` and `V`.
- `alg::String = "divide_and_conquer"`. Options:
  - `"divide_and_conquer"` - A divide-and-conquer algorithm. LAPACK's gesdd. Fast, but may lead to some innacurate singular values for very ill-conditioned matrices. Also may sometimes fail to converge, leading to errors (in which case "qr_iteration" or "recursive" can be tried).
  - `"qr_iteration"` - Typically slower but more accurate for very ill-conditioned matrices compared to `"divide_and_conquer"`. LAPACK's gesvd.
  - `"recursive"` - ITensor's custom svd. Very reliable, but may be slow if high precision is needed. To get an `svd` of a matrix `A`, an eigendecomposition of ``A^{\\dagger} A`` is used to compute `U` and then a `qr` of ``A^{\\dagger} U`` is used to compute `V`. This is performed recursively to compute small singular values.
- `use_absolute_cutoff::Bool = false`: set if all probability weights below the `cutoff` value should be discarded, rather than the sum of discarded weights.
- `use_relative_cutoff::Bool = true`: set if the singular values should be normalized for the sake of truncation.

See also: [`factorize`](@ref)
"""
function svd(A::ITensor, Linds...; kwargs...)
  utags::TagSet = get(kwargs, :lefttags, get(kwargs, :utags, "Link,u"))
  vtags::TagSet = get(kwargs, :righttags, get(kwargs, :vtags, "Link,v"))

  # Keyword argument deprecations
  #if haskey(kwargs, :utags) || haskey(kwargs, :vtags)
  #  @warn "Keyword arguments `utags` and `vtags` are deprecated in favor of `leftags` and `righttags`."
  #end

  Lis = commoninds(A, IndexSet(Linds...))
  Ris = uniqueinds(A, Lis)

  if length(Lis) == 0 || length(Ris) == 0
    error("In `svd`, the left or right indices are empty (the indices of `A` are ($(inds(A))), but the input indices are ($Lis)). For now, this is not supported. You may have accidentally input the wrong indices.")
  end

  CL = combiner(Lis...)
  CR = combiner(Ris...)

  AC = A * CR * CL

  cL = combinedind(CL)
  cR = combinedind(CR)
  if inds(AC) != IndexSet(cL, cR)
    AC = permute(AC, cL, cR)
  end

  USVT = svd(tensor(AC); kwargs...)
  if isnothing(USVT)
    return nothing
  end
  UT, ST, VT, spec = USVT
  UC, S, VC = itensor(UT), itensor(ST), itensor(VT)

  u = commonind(S,UC)
  v = commonind(S,VC)

  if hasqns(A)
    # Fix the flux of UC,S,VC
    # such that flux(UC) == flux(VC) == QN()
    # and flux(S) == flux(A)
    for b in nzblocks(UC)
      i1 = inds(UC)[1]
      i2 = inds(UC)[2]
      newqn = -dir(i2)*flux(i1 => Block(b[1]))
      setblockqn!(i2,newqn,b[2])
      setblockqn!(u,newqn,b[2])
    end

    for b in nzblocks(VC)
      i1 = inds(VC)[1]
      i2 = inds(VC)[2]
      newqn = -dir(i2)*flux(i1 => Block(b[1]))
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

ITensor factorization type for a truncated eigenvalue 
decomposition, returned by `eigen`.
"""
struct TruncEigen{N}
  D::ITensor{2}
  V::ITensor{N}
  Vt::ITensor{N}
  spec::Spectrum
  l::Index
  r::Index
end

# iteration for destructuring into components `D, V, spec, l, r = E`
iterate(E::TruncEigen) = (E.D, Val(:V))
iterate(E::TruncEigen, ::Val{:V}) = (E.V, Val(:spec))
iterate(E::TruncEigen, ::Val{:spec}) = (E.spec, Val(:l))
iterate(E::TruncEigen, ::Val{:l}) = (E.l, Val(:r))
iterate(E::TruncEigen, ::Val{:r}) = (E.r, Val(:done))
iterate(E::TruncEigen, ::Val{:done}) = nothing

function eigen(A::ITensor{N}, Linds, Rinds; kwargs...) where {N}
  @debug begin
    if hasqns(A)
      @assert flux(A) == QN()
    end
  end

  NL = length(Linds)
  NR = length(Rinds)
  NL != NR && error("Must have equal number of left and right indices")
  N != NL + NR && error("Number of left and right indices must add up to total number of indices")

  ishermitian::Bool = get(kwargs, :ishermitian, false)

  tags::TagSet = get(kwargs, :tags, "Link,eigen")
  lefttags::TagSet = get(kwargs, :lefttags, tags)
  righttags::TagSet = get(kwargs, :righttags, tags)

  plev::Int = get(kwargs, :plev, 0)
  leftplev::Int = get(kwargs, :leftplev, plev)
  rightplev::Int = get(kwargs, :rightplev, plev)

  if lefttags == righttags && leftplev == rightplev
    leftplev = rightplev + 1
  end

  # Linds, Rinds may not have the correct directions
  Lis = IndexSet(Linds...)
  Ris = IndexSet(Rinds...)

  # Ensure the indices have the correct directions,
  # QNs, etc.
  # First grab the indices in A, then permute them
  # correctly.
  Lis = permute(commoninds(A, Lis), Lis)
  Ris = permute(commoninds(A, Ris), Ris)

  for (l, r) in zip(Lis, Ris)
    if space(l) != space(r)
      error("In eigen, indices must come in pairs with equal spaces.")
    end
    if hasqns(A)
      if dir(l) == dir(r)
        error("In eigen, indices must come in pairs with opposite directions")
      end
    end
  end

  CL = combiner(Lis...; dir = Out, tags = "CMB,left")
  CR = combiner(Ris...; dir = In, tags = "CMB,right")

  AC = A * CR * CL

  cL = combinedind(CL)
  cR = combinedind(CR)
  if inds(AC) != IndexSet(cL, cR)
    AC = permute(AC, cL, cR)
  end

  AT = ishermitian ? Hermitian(tensor(AC)) : tensor(AC)

  DT, VT, spec = eigen(AT; kwargs...)
  D, VC = itensor(DT), itensor(VT)

  if hasqns(A)
    d = uniqueind(D)
    i1, i2 = inds(VC)
    for b in nzblocks(VC)
      if flux(VC, b) != QN()
        new_flux = dir(i1) * flux(i1 => Block(b[1]))
        setblockqn!(i2, new_flux, b[2])
        setblockqn!(d, new_flux, b[2])
      end
    end
  end

  V = VC * CR

  # Set right index tags
  l = uniqueind(D, V)
  r = commonind(D, V)
  l̃ = setprime(settags(l, lefttags), leftplev)
  r̃ = setprime(settags(l̃, righttags), rightplev)

  replaceinds!(D, (l, r), (l̃, r̃))
  replaceind!(V, r, r̃)
 
  l, r = l̃, r̃

  # The right eigenvectors, after being applied to A
  Vt = replaceinds(V, (Ris..., r), (Lis..., l))

  @debug begin
    if hasqns(A)
      @assert flux(D) == QN()
      @assert flux(V) == QN()
      @assert flux(Vt) == QN()
    end
  end

  return TruncEigen(D, V, Vt, spec, l, r)
end

function eigen(A::ITensor; kwargs...)
  Ris = filterinds(A; plev = 0)
  Lis = Ris'
  return eigen(A, Lis, Ris; kwargs...)
end

function qr(A::ITensor, Linds...; kwargs...)
  tags::TagSet = get(kwargs, :tags, "Link,qr")
  Lis = commoninds(A,IndexSet(Linds...))
  Ris = uniqueinds(A,Lis)
  Lpos,Rpos = NDTensors.getperms(inds(A),Lis,Ris)
  QT,RT = qr(tensor(A),Lpos,Rpos;kwargs...)
  Q,R = itensor(QT),itensor(RT)
  q = commonind(Q,R)
  settags!(Q,tags,q)
  settags!(R,tags,q)
  q = settags(q,tags)
  return Q,R,q
end

# TODO: allow custom tags in internal indices?
function polar(A::ITensor, Linds...; kwargs...)
  Lis = commoninds(A, IndexSet(Linds...))
  Ris = uniqueinds(A, Lis)
  Lpos, Rpos = NDTensors.getperms(inds(A), Lis, Ris)
  UT, PT = polar(tensor(A), Lpos, Rpos)
  U, P = itensor(UT), itensor(PT)
  u = commoninds(U, P)
  p = uniqueinds(P, U)
  replaceinds!(U, u, p')
  replaceinds!(P, u, p')
  return U, P, commoninds(U, P)
end

function factorize_qr(A::ITensor, Linds...; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  if ortho == "left"
    L,R,q = qr(A,Linds...; kwargs...)
  elseif ortho == "right"
    Lis = uniqueinds(A,IndexSet(Linds...))
    R,L,q = qr(A,Lis...; kwargs...)
  else
    error("In factorize using qr decomposition, ortho keyword $ortho not supported. Supported options are left or right.")
  end
  return L,R
end

function factorize_svd(A::ITensor, Linds...; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  USV = svd(A, Linds...; kwargs..., alg = alg)
  if isnothing(USV)
    return nothing
  end
  U, S, V, spec, u, v = USV
  if ortho == "left"
    L, R = U, S * V
  elseif ortho == "right"
    L,R = U * S, V
  elseif ortho == "none"
    sqrtS = S
    sqrtS .= sqrt.(S)
    L, R = U * sqrtS, sqrtS * V
    replaceind!(L, v, u)
  else
    error("In factorize using svd decomposition, ortho keyword $ortho not supported. Supported options are left, right, or none.")
  end
  return L, R, spec
end

function factorize_eigen(A::ITensor, Linds...; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  delta_A2 = get(kwargs, :eigen_perturbation, nothing)
  if ortho == "left"
    Lis = commoninds(A, IndexSet(Linds...))
  elseif ortho == "right"
    Lis = uniqueinds(A, IndexSet(Linds...))
  else
    error("In factorize using eigen decomposition, ortho keyword $ortho not supported. Supported options are left or right.")
  end
  simLis = sim(Lis)
  A2 = A * replaceinds(dag(A), Lis, simLis)
  if !isnothing(delta_A2)
    # This assumes delta_A2 has indices:
    # (Lis..., prime(Lis)...)
    A2 += replaceinds(delta_A2, prime(Lis), simLis)
  end
  F = eigen(A2, Lis, simLis; ishermitian=true,
                             kwargs...)
  D, _, spec = F
  L = F.Vt
  R = dag(L) * A
  if ortho == "right"
    L, R = R, L
  end
  return L, R, spec
end

"""
    factorize(A::ITensor, Linds::Index...; <keyword arguments>)

Perform a factorization of `A` into ITensors `L` and `R` such that `A ≈ L * R`.

# Arguments
- `ortho::String = "left"`: Choose orthogonality properties of the factorization.
  - `"left"`: the left factor `L` is an orthogonal basis such that `L * dag(prime(L, commonind(L,R))) ≈ I`. 
  - `"right"`: the right factor `R` forms an orthogonal basis. 
  - `"none"`, neither of the factors form an orthogonal basis, and in general are made as symmetrically as possible (depending on the decomposition used).
- `which_decomp::Union{String, Nothing} = nothing`: choose what kind of decomposition is used. 
  - `nothing`: choose the decomposition automatically based on the other arguments. For example, when `nothing` is chosen and `ortho = "left"` or `"right"`, and a cutoff is provided, `svd` or `eigen` is used depending on the provided cutoff (`eigen` is only used when the cutoff is greater than `1e-12`, since it has a lower precision). When no truncation is requested `qr` is used for dense ITensors and `svd` for block-sparse ITensors (in the future `qr` will be used also for block-sparse ITensors in this case).
  - `"svd"`: `L = U` and `R = S * V` for `ortho = "left"`, `L = U * S` and `R = V` for `ortho = "right"`, and `L = U * sqrt.(S)` and `R = sqrt.(S) * V` for `ortho = "none"`. To control which `svd` algorithm is choose, use the `svd_alg` keyword argument. See the documentation for `svd` for the supported algorithms, which are the same as those accepted by the `alg` keyword argument.
  - `"eigen"`: `L = U` and ``R = U^{\\dagger} A`` where `U` is determined from the eigendecompositon ``A A^{\\dagger} = U D U^{\\dagger}`` for `ortho = "left"` (and vice versa for `ortho = "right"`). `"eigen"` is not supported for `ortho = "none"`.
  - `"qr"`: `L=Q` and `R` an upper-triangular matrix when `ortho = "left"`, and `R = Q` and `L` a lower-triangular matrix when `ortho = "right"` (currently supported for dense ITensors only).
In the future, other decompositions like QR (for block-sparse ITensors), polar, cholesky, LU, etc. are expected to be supported.

For truncation arguments, see: [`svd`](@ref)
"""
function factorize(A::ITensor, Linds...; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  tags::TagSet = get(kwargs, :tags, "Link,fact")
  plev::Int = get(kwargs, :plev, 0)
  which_decomp::Union{String, Nothing} = get(kwargs, :which_decomp, nothing)
  cutoff = get(kwargs, :cutoff, nothing)
  eigen_perturbation = get(kwargs, :eigen_perturbation, nothing)
  if !isnothing(eigen_perturbation)
    if !(isnothing(which_decomp) || which_decomp == "eigen")
      error("""when passing a non-trivial eigen_perturbation to `factorize`,
               the which_decomp keyword argument must be either "automatic" or
               "eigen" """)
    end
    which_decomp = "eigen"
  end

  # Deprecated keywords
  if haskey(kwargs, :dir)
    error("""dir keyword in factorize has been replace by ortho.
    Note that the default is now `left`, meaning for the results L,R = factorize(A), L forms an orthogonal basis.""")
  end

  if haskey(kwargs, :which_factorization)
    error("""which_factorization keyword in factorize has been replace by which_decomp.""")
  end

  # Determines when to use eigen vs. svd (eigen is less precise,
  # so eigen should only be used if a larger cutoff is requested)
  automatic_cutoff = 1e-12

  dL,dR = dim(IndexSet(Linds...)), dim(IndexSet(setdiff(inds(A),Linds)...))
  maxdim = get(kwargs,:maxdim, min(dL, dR))
  might_truncate = !isnothing(cutoff) || maxdim < min(dL, dR)

  if isnothing(which_decomp)
    if !might_truncate && !hasqns(A) && ortho != "none"
      which_decomp="qr"
    elseif isnothing(cutoff) || cutoff ≤ automatic_cutoff
      which_decomp="svd"
    elseif cutoff > automatic_cutoff
      which_decomp="eigen"
    end
  end

  if which_decomp == "svd"
    LR = factorize_svd(A, Linds...; kwargs...)
    if isnothing(LR)
      return nothing
    end
    L, R, spec = LR
  elseif which_decomp == "eigen"
    L, R, spec = factorize_eigen(A, Linds...; kwargs...)
  elseif which_decomp == "qr"
    hasqns(A) && error("QR factorization of an ITensor with QNs is not yet supported.")
    L,R = factorize_qr(A,Linds...; kwargs...)
    spec = Spectrum(nothing,0.0)
  else
    throw(ArgumentError("""In factorize, factorization $which_decomp is not currently supported. Use `"svd"`, `"eigen"`, `"qr"` or `nothing`."""))
  end

  # Set the tags and prime level
  l = commonind(L, R)
  l̃ = setprime(settags(l, tags), plev)
  replaceind!(L, l, l̃)
  replaceind!(R, l, l̃)
  l = l̃

  return L, R, spec, l
end

