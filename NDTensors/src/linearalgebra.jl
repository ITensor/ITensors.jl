export eigs,
       entropy,
       polar,
       random_orthog,
       Spectrum,
       svd,
       truncerror

#
# Linear Algebra of order 2 NDTensors
#
# Even though DenseTensor{_,2} is strided
# and passable to BLAS/LAPACK, it cannot
# be made <: StridedArray

function Base.:*(T1::Tensor{ElT1,2,StoreT1},
                 T2::Tensor{ElT2,2,StoreT2}) where
                                       {ElT1,StoreT1<:Dense,IndsT1,
                                        ElT2,StoreT2<:Dense,IndsT2}
  RM = matrix(T1)*matrix(T2)
  indsR = (ind(T1,1), ind(T2,2))
  return tensor(Dense(vec(RM)), indsR)
end

function LinearAlgebra.exp(T::DenseTensor{ElT,2}) where {ElT}
  expTM = exp(matrix(T))
  return tensor(Dense(vec(expTM)),inds(T))
end

function LinearAlgebra.exp(T::Hermitian{ElT,
                                        <:DenseTensor{ElT,2}}) where {ElT<:Union{Real,Complex}}
  # exp(::Hermitian/Symmetric) returns Hermitian/Symmetric,
  # so extract the parent matrix
  expTM = parent(exp(matrix(T)))
  return tensor(Dense(vec(expTM)),inds(T))
end

"""
  Spectrum
contains the (truncated) density matrix eigenvalue spectrum which is computed during a
decomposition done by `svd` or `eigen`. In addition stores the truncation error.
"""
struct Spectrum{VecT<:AbstractVector}
  eigs::VecT
  truncerr::Float64
end

eigs(s::Spectrum) = s.eigs
truncerror(s::Spectrum) = s.truncerr

function entropy(s::Spectrum)
  S = 0.0
  for p in eigs(s)
    p > 1e-13 && (S -= p*log(p))
  end
  return S
end

"""
    svd(T::DenseTensor{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function LinearAlgebra.svd(T::DenseTensor{ElT,2,IndsT};
                           kwargs...) where {ElT,IndsT}
  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  #
  # Keyword argument deprecations
  #
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs,
                              :absoluteCutoff,
                              use_absolute_cutoff)
  end

  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs,
                              :doRelCutoff,
                              use_relative_cutoff)
  end

  if haskey(kwargs, :fastsvd) || haskey(kwargs, :fastSVD)
    error("In svd, fastsvd/fastSVD keyword arguments are removed in favor of alg, see documentation for more details.")
  end

  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  use_absolute_cutoff::Bool = get(kwargs,
                                  :use_absolute_cutoff,
                                  use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs,
                                  :use_relative_cutoff,
                                  use_relative_cutoff)
  alg::String = get(kwargs, :alg, "recursive")

  if alg == "divide_and_conquer"
    MU,MS,MV = svd(matrix(T); alg = LinearAlgebra.DivideAndConquer())
  elseif alg == "qr_iteration"
    MU,MS,MV = svd(matrix(T); alg = LinearAlgebra.QRIteration())
  elseif alg == "recursive"
    MU,MS,MV = svd_recursive(matrix(T))
  else
    error("svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.")
  end
  conj!(MV)

  P = MS .^ 2
  if truncate
    truncerr, _ = truncate!(P; mindim = mindim,
                               maxdim = maxdim,
                               cutoff = cutoff,
                               use_absolute_cutoff = use_absolute_cutoff,
                               use_relative_cutoff = use_relative_cutoff)
  else
    truncerr = 0.0
  end
  spec = Spectrum(P, truncerr)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    resize!(MS,dS)
    MV = MV[:,1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T,1),u))
  Sinds = IndsT((u,v))
  Vinds = IndsT((ind(T,2),v))
  U = tensor(Dense(vec(MU)),Uinds)
  S = tensor(Diag(MS),Sinds)
  V = tensor(Dense(vec(MV)),Vinds)
  return U,S,V,spec
end

function LinearAlgebra.eigen(T::Hermitian{ElT,<:DenseTensor{ElT,2,IndsT}};
                             kwargs...) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs,
                              :absoluteCutoff,
                              use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs,
                              :doRelCutoff,
                              use_relative_cutoff)
  end

  truncate = haskey(kwargs,:maxdim) || haskey(kwargs,:cutoff)
  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  use_absolute_cutoff::Bool = get(kwargs,
                                  :use_absolute_cutoff,
                                  use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs,
                                  :use_relative_cutoff,
                                  use_relative_cutoff)

  DM, VM = eigen(matrix(T))

  # Sort by largest to smallest eigenvalues
  p = sortperm(DM; rev = true)
  DM = DM[p]
  VM = VM[:,p]

  if truncate
    truncerr,_ = truncate!(DM;maxdim=maxdim,
                              cutoff=cutoff,
                              use_absolute_cutoff=use_absolute_cutoff,
                              use_relative_cutoff=use_relative_cutoff)
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(DM,truncerr)

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
    random_orthog(n::Int,m::Int)

Return a random matrix O of dimensions (n,m)
such that if n >= m, transpose(O)*O is the 
identity, or if m > n O*transpose(O) is the
identity.
"""
function random_orthog(n::Int,m::Int)::Matrix
  #TODO: optimize by using recursive Householder algorithm?
  if n < m
    return transpose(random_orthog(m,n))
  end
  F = qr(randn(n,m))
  Q = Matrix(F.Q)
  for c=1:size(Q,2)
    if real(F.R[c,c]) < 0.0
      Q[:,c] *= -1
    end
  end
  return Q
end

function qr_positive(M::AbstractMatrix)
  sparseQ,R = qr(M)
  Q = convert(Matrix,sparseQ)
  nc = size(Q,2)
  for c=1:nc
    if real(R[c,c]) < 0.0
      R[c,c:end] *= -1
      Q[:,c] *= -1
    end
  end
  return (Q,R)
end

function LinearAlgebra.eigen(T::DenseTensor{ElT,2,IndsT};
                             kwargs...) where {ElT<:Union{Real,Complex},IndsT}
  # Keyword argument deprecations
  use_absolute_cutoff = false
  if haskey(kwargs, :absoluteCutoff)
    @warn "In svd, keyword argument absoluteCutoff is deprecated in favor of use_absolute_cutoff"
    use_absolute_cutoff = get(kwargs,
                              :absoluteCutoff,
                              use_absolute_cutoff)
  end
  use_relative_cutoff = true
  if haskey(kwargs, :doRelCutoff)
    @warn "In svd, keyword argument doRelCutoff is deprecated in favor of use_relative_cutoff"
    use_relative_cutoff = get(kwargs,
                              :doRelCutoff,
                              use_relative_cutoff)
  end

  truncate = haskey(kwargs,:maxdim) || haskey(kwargs,:cutoff)
  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  use_absolute_cutoff::Bool = get(kwargs,
                                  :use_absolute_cutoff,
                                  use_absolute_cutoff)
  use_relative_cutoff::Bool = get(kwargs,
                                  :use_relative_cutoff,
                                  use_relative_cutoff)

  DM, VM = eigen(matrix(T))

  # Sort by largest to smallest eigenvalues
  #p = sortperm(DM; rev = true)
  #DM = DM[p]
  #VM = VM[:,p]

  if truncate
    truncerr,_ = truncate!(DM;maxdim=maxdim,
                              cutoff=cutoff,
                              use_absolute_cutoff=use_absolute_cutoff,
                              use_relative_cutoff=use_relative_cutoff)
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

function LinearAlgebra.qr(T::DenseTensor{ElT,2,IndsT}
                          ;kwargs...) where {ElT,IndsT}
  positive = get(kwargs,:positive,false)
  # TODO: just call qr on T directly (make sure
  # that is fast)
  if positive
    QM,RM = qr_positive(matrix(T))
  else
    QM,RM = qr(matrix(T))
  end
  # Make the new indices to go onto Q and R
  q,r = inds(T)
  q = dim(q) < dim(r) ? sim(q) : sim(r)
  Qinds = IndsT((ind(T,1),q))
  Rinds = IndsT((q,ind(T,2)))
  Q = tensor(Dense(vec(Matrix(QM))),Qinds)
  R = tensor(Dense(vec(RM)),Rinds)
  return Q,R
end

# TODO: support alg keyword argument to choose the svd algorithm
function polar(T::DenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
  QM,RM = polar(matrix(T))
  dim = size(QM,2)
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  # TODO: use push/pushfirst instead of a constructor
  # call here
  Qinds = IndsT((ind(T,1),q))
  Rinds = IndsT((q,ind(T,2)))
  Q = tensor(Dense(vec(QM)),Qinds)
  R = tensor(Dense(vec(RM)),Rinds)
  return Q,R
end

