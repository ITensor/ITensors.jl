export eigs, entropy, polar, random_orthog, random_unitary, Spectrum, svd, truncerror

#
# Linear Algebra of order 2 NDTensors
#
# Even though DenseTensor{_,2} is strided
# and passable to BLAS/LAPACK, it cannot
# be made <: StridedArray

function (
  T1::Tensor{ElT1,2,StoreT1} * T2::Tensor{ElT2,2,StoreT2}
) where {ElT1,StoreT1<:Dense,ElT2,StoreT2<:Dense}
  RM = matrix(T1) * matrix(T2)
  indsR = (ind(T1, 1), ind(T2, 2))
  return tensor(Dense(vec(RM)), indsR)
end

function LinearAlgebra.exp(T::DenseTensor{ElT,2}) where {ElT<:Union{Real,Complex}}
  expTM = exp(matrix(T))
  return tensor(Dense(vec(expTM)), inds(T))
end

function LinearAlgebra.exp(
  T::Hermitian{ElT,<:DenseTensor{ElT,2}}
) where {ElT<:Union{Real,Complex}}
  # exp(::Hermitian/Symmetric) returns Hermitian/Symmetric,
  # so extract the parent matrix
  expTM = parent(exp(matrix(T)))
  return tensor(Dense(vec(expTM)), inds(T))
end

"""
  Spectrum
contains the (truncated) density matrix eigenvalue spectrum which is computed during a
decomposition done by `svd` or `eigen`. In addition stores the truncation error.
"""
struct Spectrum{VecT<:Union{AbstractVector,Nothing},ElT<:Real}
  eigs::VecT
  truncerr::ElT
end

eigs(s::Spectrum) = s.eigs
truncerror(s::Spectrum) = s.truncerr

function entropy(s::Spectrum)
  S = 0.0
  eigs_s = eigs(s)
  isnothing(eigs_s) &&
    error("Spectrum does not contain any eigenvalues, cannot compute the entropy")
  for p in eigs_s
    p > 1e-13 && (S -= p * log(p))
  end
  return S
end

function svd_catch_error(A; kwargs...)
  USV = try
    svd(A; kwargs...)
  catch
    return nothing
  end
  return USV
end

function lapack_svd_error_message(alg)
  return "The SVD algorithm `\"$alg\"` has thrown an error,\n" *
         "likely because of a convergance failure. You can try\n" *
         "other SVD algorithms that may converge better using the\n" *
         "`alg` (or `svd_alg` if called through `factorize` or MPS/MPO functionality) keyword argument:\n\n" *
         " - \"divide_and_conquer\" is a divide-and-conquer algorithm\n" *
         "   (LAPACK's `gesdd`). It is fast, but may lead to some innacurate\n" *
         "   singular values for very ill-conditioned matrices.\n" *
         "   It also may sometimes fail to converge, leading to errors\n" *
         "   (in which case `\"qr_iteration\"` or `\"recursive\"` can be tried).\n\n" *
         " - `\"qr_iteration\"` (LAPACK's `gesvd`) is typically slower \n" *
         "   than \"divide_and_conquer\", especially for large matrices,\n" *
         "   but is more accurate for very ill-conditioned matrices \n" *
         "   compared to `\"divide_and_conquer\"`.\n\n" *
         " - `\"recursive\"` is ITensor's custom SVD algorithm. It is very\n" *
         "   reliable, but may be slow if high precision is needed.\n" *
         "   To get an `svd` of a matrix `A`, an eigendecomposition of\n" *
         "   ``A^{\\dagger} A`` is used to compute `U` and then a `qr` of\n" *
         "   ``A^{\\dagger} U`` is used to compute `V`. This is performed\n" *
         "   recursively to compute small singular values.\n\n" *
         "Returning `nothing`. For an output `F = svd(A, ...)` you can check if\n" *
         "`isnothing(F)` in your code and try a different algorithm.\n\n" *
         "To suppress this message in the future, you can wrap the `svd` call in the\n" *
         "`@suppress` macro from the `Suppressor` package.\n"
end

"""
    svd(T::DenseTensor{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function LinearAlgebra.svd(T::DenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  #
  # Keyword argument deprecations
  #
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end

  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  if haskey(kwargs, :fastsvd) || haskey(kwargs, :fastSVD)
    error(
      "In svd, fastsvd/fastSVD keyword arguments are removed in favor of alg, see documentation for more details.",
    )
  end

  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)
  alg::String = get(kwargs, :alg, "divide_and_conquer")

  #@timeit_debug timer "dense svd" begin
  if alg == "divide_and_conquer"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.DivideAndConquer())
    if isnothing(MUSV)
      # If "divide_and_conquer" fails, try "qr_iteration"
      alg = "qr_iteration"
      MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
      if isnothing(MUSV)
        # If "qr_iteration" fails, try "recursive"
        alg = "recursive"
        MUSV = svd_recursive(matrix(T))
      end
    end
  elseif alg == "qr_iteration"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
    if isnothing(MUSV)
      # If "qr_iteration" fails, try "recursive"
      alg = "recursive"
      MUSV = svd_recursive(matrix(T))
    end
  elseif alg == "recursive"
    MUSV = svd_recursive(matrix(T))
  else
    error(
      "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
    )
  end
  if isnothing(MUSV)
    if any(isnan, T)
      println("SVD failed, the matrix you were trying to SVD contains NaNs.")
    else
      println(lapack_svd_error_message(alg))
    end
    return nothing
  end
  MU, MS, MV = MUSV
  conj!(MV)
  #end # @timeit_debug

  P = MS .^ 2
  if truncate
    truncerr, _ = truncate!(
      P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
  else
    truncerr = 0.0
  end
  spec = Spectrum(P, truncerr)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:, 1:dS]
    resize!(MS, dS)
    MV = MV[:, 1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T, 1), u))
  Sinds = IndsT((u, v))
  Vinds = IndsT((ind(T, 2), v))
  U = tensor(Dense(vec(MU)), Uinds)
  S = tensor(Diag(MS), Sinds)
  V = tensor(Dense(vec(MV)), Vinds)
  return U, S, V, spec
end

function LinearAlgebra.eigen(
  T::Hermitian{ElT,<:DenseTensor{ElT,2,IndsT}}; kwargs...
) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)
  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff::Union{Nothing,Float64} = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  DM, VM = eigen(matrixT)

  # Sort by largest to smallest eigenvalues
  p = sortperm(DM; rev=true, by=abs)
  DM = DM[p]
  VM = VM[:, p]

  if truncate
    truncerr, _ = truncate!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(DM, truncerr)

  # Make the new indices to go onto V
  l = eltype(IndsT)(dD)
  r = eltype(IndsT)(dD)
  Vinds = IndsT((dag(ind(T, 2)), dag(r)))
  Dinds = IndsT((l, dag(r)))
  V = tensor(Dense(vec(VM)), Vinds)
  D = tensor(Diag(DM), Dinds)
  return D, V, spec
end

"""
    random_unitary(n::Int,m::Int)::Matrix{ComplexF64}
    random_unitary(::Type{ElT},n::Int,m::Int)::Matrix{ElT}

Return a random matrix U of dimensions (n,m)
such that if n >= m, U'*U is the identity, or if
m > n U*U' is the identity. Optionally can pass a numeric
type as the first argument to obtain a matrix of that type.

Sampling is based on https://arxiv.org/abs/math-ph/0609050
such that in the case `n==m`, the unitary matrix will be sampled
according to the Haar measure.
"""
function random_unitary(::Type{ElT}, n::Int, m::Int) where {ElT<:Number}
  return random_unitary(Random.default_rng(), ElT, n, m)
end

function random_unitary(rng::AbstractRNG, ::Type{ElT}, n::Int, m::Int) where {ElT<:Number}
  if n < m
    return Matrix(random_unitary(rng, ElT, m, n)')
  end
  F = qr(randn(rng, ElT, n, m))
  Q = Matrix(F.Q)
  # The upper triangle of F.factors 
  # are the elements of R.
  # Multiply cols of Q by the signs
  # that would make diagonal of R 
  # non-negative:
  for c in 1:size(Q, 2)
    Q[:, c] .*= sign(F.factors[c, c])
  end
  return Q
end

random_unitary(n::Int, m::Int) = random_unitary(ComplexF64, n, m)

"""
    random_orthog(n::Int,m::Int)::Matrix{Float64}
    random_orthog(::Type{ElT},n::Int,m::Int)::Matrix{ElT}

Return a random, real matrix O of dimensions (n,m)
such that if n >= m, transpose(O)*O is the
identity, or if m > n O*transpose(O) is the
identity. Optionally can pass a real number type
as the first argument to obtain a matrix of that type.
"""
random_orthog(::Type{ElT}, n::Int, m::Int) where {ElT<:Real} = random_unitary(ElT, n, m)

random_orthog(n::Int, m::Int) = random_orthog(Float64, n, m)

function LinearAlgebra.eigen(
  T::DenseTensor{ElT,2,IndsT}; kwargs...
) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs, :absoluteCutoff, use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs, :doRelCutoff, use_relative_cutoff)
  end

  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)
  maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
  mindim::Int = get(kwargs, :mindim, 1)
  cutoff::Float64 = get(kwargs, :cutoff, 0.0)
  use_absolute_cutoff::Bool = get(kwargs, :use_absolute_cutoff, use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs, :use_relative_cutoff, use_relative_cutoff)

  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  DM, VM = eigen(matrixT)

  # Sort by largest to smallest eigenvalues
  #p = sortperm(DM; rev = true)
  #DM = DM[p]
  #VM = VM[:,p]

  if truncate
    truncerr, _ = truncate!(
      DM; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, kwargs...
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(abs.(DM), truncerr)

  i1, i2 = inds(T)

  # Make the new indices to go onto D and V
  l = typeof(i1)(dD)
  r = dag(sim(l))
  Dinds = (l, r)
  Vinds = (dag(i2), r)
  D = complex(tensor(Diag(DM), Dinds))
  V = complex(tensor(Dense(vec(VM)), Vinds))
  return D, V, spec
end
#
#  Trim out n rows of R based on norm(R_nn)<cutoff, where R_nn is bottom n rows of R. 
#  Also trim the corresponding columns of Q. 
#
function trim_rows(Q::AbstractMatrix, R::AbstractMatrix, atol::Float64, rtol::Float64; verbose=false)
  nr = size(R, 1)
  @assert size(Q, 2) == nr #Sanity check.
  #
  #  Find the largest n such than norm(R_nn)<=cutoff, where Rnn if the bottom right block with rows
  #  from n:nr.  n=last_row_to_keep+1
  #
  last_row_to_keep = nr
  do_atol,do_rtol = atol>=0.0,rtol>=0.0
  # for r in nr:-1:1
  #   Rnn=norm(R[r:nr, :])
  #   R11=norm(R[1:r-1, :])
  #   if (do_atol && Rnn > atol) || (do_rtol && Rnn/R11 > rtol)
  #     last_row_to_keep = r
  #     break
  #   end
  # end
  #
  #  Could also do the same test but only looking at the diagonals
  #
  dR=diag(R)
  for r in nr:-1:1
    Rnn=norm(dR[r:nr])
    R11=norm(dR[1:r-1])
    if (do_atol && Rnn > atol) || (do_rtol && Rnn/R11 > rtol)
      last_row_to_keep = r
      break
    end
  end

  num_zero_rows = nr - last_row_to_keep
  if num_zero_rows == 0
    verbose && println("Rank Reveal removing $num_zero_rows rows with atol=$atol, rtol=$rtol")
    return Q, R
  end
  #
  # Useful output for trouble shooting.
  #
  if verbose
    println("Rank Reveal removing $num_zero_rows rows with atol=$atol, rtol=$rtol")
  end

  return Q[:, 1:last_row_to_keep], R[1:last_row_to_keep, :]
end

if VERSION >= v"1.7"
  struct RowNorm end #for row pivoting lq
end

qr(T::DenseTensor{<:Any,2}; kwargs...) = qx(qr, T; kwargs...)
ql(T::DenseTensor{<:Any,2}; kwargs...) = qx(ql, T; kwargs...)

translate_pivot(pivot::Bool)::Bool=pivot
if VERSION >= v"1.7"
  translate_pivot(pivot::NoPivot)::Bool=false
  translate_pivot(pivot::ColumnNorm)::Bool=true
  translate_pivot(pivot::RowNorm)::Bool=true
end

#
#  Generic function for qr and ql decomposition of dense matrix.
#  The X tensor = R or L.
#
function qx(
  qx::Function,
  T::DenseTensor{<:Any,2};
  positive=false,
  pivot=false,
  atol=-1.0, #absolute tolerance for rank reduction
  rtol=-1.0, #relative tolerance for rank reduction
  block_rtol=-1.0, #This is supposed to be for block sparse, but we reluctantly accept it here.
  verbose=false,
  kwargs...,
)

  pivot=translate_pivot(pivot)
 
  if rtol<0.0 && block_rtol>=0.0
    rtol=block_rtol
  end
  do_rank_reduction = (atol >= 0.0) || (rtol >= 0.0)
  if do_rank_reduction && qx == ql
    @warn "User requested rq/ql decomposition with atol=$atol, rtol=$rtol." *
      "  Rank reduction requires column pivoting which is not supported for rq/ql decomposition in lapack/ITensors"
    do_rank_reduction = false
  end
  if pivot && qx == ql
    @warn "User requested rq/ql decomposition with column pivoting." *
      "  Column pivoting is not supported for rq/ql decomposition in lapack/ITensors"
    pivot = false
  end
  if do_rank_reduction
    pivot = true
  end

  if pivot
    QM, XM, p = qx(matrix(T), Val(true)) #with colun pivoting
    QM, XM = trim_rows(Matrix(QM), XM, atol, rtol; verbose=verbose)
  else
    QM, XM = qx(matrix(T), Val(false)) #no column pivoting
    QM = Matrix(QM)
    p = nothing
  end
  #
  #  Gauge fix diagonal of X into positive definite form. 
  #
  positive && qx_positive!(qx, QM, XM)
  #
  #  undo the permutation on R, so the T=Q*R again.
  #
  pivot && (XM = XM[:, invperm(p)])
  #
  # Make the new indices to go onto Q and X
  #
  IndsT = indstype(T) #get the indices type
  @assert IndsT.parameters[1] == IndsT.parameters[2] #they better be the same!
  IndexT = IndsT.parameters[1] #establish the single index type.
  q = IndexT(size(XM)[1]) #create the Q--X link index.
  Qinds = IndsT((ind(T, 1), q))
  Xinds = IndsT((q, ind(T, 2)))
  Q = tensor(Dense(vec(QM)), Qinds)
  X = tensor(Dense(vec(XM)), Xinds)
  return Q, X, p
end

# Required by svd_recursive 
"""
    qr_positive(M::AbstractMatrix)

Compute the QR decomposition of a matrix M
such that the diagonal elements of R are
non-negative. Such a QR decomposition of a
matrix is unique. Returns a tuple (Q,R).
"""
function qr_positive(M::AbstractMatrix)
  sparseQ, R = qr(M)
  Q = convert(Matrix, sparseQ)
  nc = size(Q, 2)
  for c in 1:nc
    if R[c, c] != 0.0 #sign(0.0)==0.0 so we don't want to zero out a column of Q.
      sign_Rc = sign(R[c, c])
      if !isone(sign_Rc)
        R[c, c:end] *= conj(sign_Rc) #only fip non-zero portion of the row.
        Q[:, c] *= sign_Rc
      end
    end
  end
  return (Q, R)
end

#
#  Semi generic function for gauge fixing the diagonal of X into positive definite form.
#  becuase the diagonal is difficult to locate for rectangular X (it moves between R and L)
#  we use qx==ql to know if X is lower or upper.
#
function qx_positive!(qx::Function, Q::AbstractMatrix, X::AbstractMatrix)
  nr, nc = size(X)
  dc = (nc > nr && qx == ql) ? nc - nr : 0 #diag is shifted over by dc if nc>nr
  for c in 1:Base.min(nr, nc)
    if X[c, c + dc] != 0.0 #sign(0.0)==0.0 so we don't want to zero out a column of Q.
      sign_Xc = sign(X[c, c + dc])
      if !isone(sign_Xc)
        X[c, :] *= sign_Xc
        Q[:, c] *= conj(sign_Xc)
      end
    end
  end
end

#
#  Lapack replaces A with Q & L carefully packed together.  So here we just copy a
#  before letting lapack overwirte it. 
#
function ql(A::AbstractMatrix, pivot; kwargs...)
  @assert pivot == Val(false)
  Base.require_one_based_indexing(A)
  T = eltype(A)
  AA = similar(A, LinearAlgebra._qreltype(T), size(A))
  copyto!(AA, A)
  return ql!(AA; kwargs...)
end
#
# This is where the low level call to lapack actually occurs.  Most of the work is
# about unpacking Q and L from the A matrix.
#
function ql!(A::StridedMatrix{<:LAPACK.BlasFloat})
  tau = Base.similar(A, min(size(A)...))
  x = LAPACK.geqlf!(A, tau)
  #save L from the lower portion of A, before orgql! mangles it!
  nr, nc = size(A)
  mn = min(nr, nc)
  L = similar(A, (mn, nc))
  for r in 1:mn
    for c in 1:(r + nc - mn)
      L[r, c] = A[r + nr - mn, c]
    end
    for c in (r + 1 + nc - mn):nc
      L[r, c] = 0.0
    end
  end
  # Now we need shift the orth vectors from the right side of Q over the left side, before
  if (mn < nc)
    for r in 1:nr
      for c in 1:mn
        A[r, c] = A[r, c + nc - mn]
      end
    end
    for r in 1:nr
      A = A[:, 1:mn] #whack the extra columns in A.
    end
  end
  LAPACK.orgql!(A, tau)
  return A, L
end

# TODO: support alg keyword argument to choose the svd algorithm
function polar(T::DenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
  QM, RM = polar(matrix(T))
  dim = size(QM, 2)
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  # TODO: use push/pushfirst instead of a constructor
  # call here
  Qinds = IndsT((ind(T, 1), q))
  Rinds = IndsT((q, ind(T, 2)))
  Q = tensor(Dense(vec(QM)), Qinds)
  R = tensor(Dense(vec(RM)), Rinds)
  return Q, R
end
