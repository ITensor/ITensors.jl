
const BlockSparseMatrix{ElT,StoreT,IndsT} = BlockSparseTensor{ElT,2,StoreT,IndsT}
const DiagBlockSparseMatrix{ElT,StoreT,IndsT} = DiagBlockSparseTensor{ElT,2,StoreT,IndsT}
const DiagMatrix{ElT,StoreT,IndsT} = DiagTensor{ElT,2,StoreT,IndsT}

function _truncated_blockdim(
  S::DiagMatrix, docut::Real; singular_values=false, truncate=true, min_blockdim=nothing
)
  min_blockdim = replace_nothing(min_blockdim, 0)
  # TODO: Replace `cpu` with `leaf_parenttype` dispatch.
  S = cpu(S)
  full_dim = diaglength(S)
  !truncate && return full_dim
  min_blockdim = min(min_blockdim, full_dim)
  newdim = 0
  val = singular_values ? getdiagindex(S, newdim + 1)^2 : abs(getdiagindex(S, newdim + 1))
  while newdim + 1 ≤ full_dim && val > docut
    newdim += 1
    if newdim + 1 ≤ full_dim
      val =
        singular_values ? getdiagindex(S, newdim + 1)^2 : abs(getdiagindex(S, newdim + 1))
    end
  end
  (newdim >= min_blockdim) || (newdim = min_blockdim)
  return newdim
end

"""
    svd(T::BlockSparseTensor{<:Number,2}; kwargs...)

svd of an order-2 BlockSparseTensor.

This function assumes that there is one block
per row/column, otherwise it fails.
This assumption makes it so the result can be
computed from the dense svds of seperate blocks.
"""
function svd(
  T::Tensor{ElT,2,<:BlockSparse};
  min_blockdim=nothing,
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {ElT}
  Us = Vector{DenseTensor{ElT,2}}(undef, nnzblocks(T))
  Ss = Vector{DiagTensor{real(ElT),2}}(undef, nnzblocks(T))
  Vs = Vector{DenseTensor{ElT,2}}(undef, nnzblocks(T))

  # Sorted eigenvalues
  d = Vector{real(ElT)}()

  for (n, b) in enumerate(eachnzblock(T))
    blockT = blockview(T, b)
    USVb = svd(blockT; alg)
    if isnothing(USVb)
      return nothing
    end
    Ub, Sb, Vb = USVb
    Us[n] = Ub
    Ss[n] = Sb
    Vs[n] = Vb
    # Previously this was:
    # vector(diag(Sb))
    # But it broke, did `diag(::Tensor)` change types?
    # TODO: call this a function `diagonal`, i.e.:
    # https://github.com/JuliaLang/julia/issues/30250
    # or make `diag(::Tensor)` return a view by default.
    append!(d, data(Sb))
  end

  # Square the singular values to get
  # the eigenvalues
  d .= d .^ 2
  sort!(d; rev=true)

  # Get the list of blocks of T
  # that are not dropped
  nzblocksT = nzblocks(T)

  dropblocks = Int[]
  if any(!isnothing, (maxdim, cutoff))
    d, truncerr, docut = truncate!!(
      d; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    for n in 1:nnzblocks(T)
      blockdim = _truncated_blockdim(
        Ss[n], docut; min_blockdim, singular_values=true, truncate=true
      )
      if blockdim == 0
        push!(dropblocks, n)
      else
        # TODO: Replace call to `data` with `diagview`.
        Strunc = tensor(Diag(data(Ss[n])[1:blockdim]), (blockdim, blockdim))
        Us[n] = Us[n][1:dim(Us[n], 1), 1:blockdim]
        Ss[n] = Strunc
        Vs[n] = Vs[n][1:dim(Vs[n], 1), 1:blockdim]
      end
    end
    deleteat!(Us, dropblocks)
    deleteat!(Ss, dropblocks)
    deleteat!(Vs, dropblocks)
    deleteat!(nzblocksT, dropblocks)
  else
    truncerr, docut = 0.0, 0.0
  end

  # The number of non-zero blocks of T remaining
  nnzblocksT = length(nzblocksT)

  #
  # Make indices of U and V 
  # that connect to S
  #
  i1 = ind(T, 1)
  i2 = ind(T, 2)
  uind = dag(sim(i1))
  vind = dag(sim(i2))
  resize!(uind, nnzblocksT)
  resize!(vind, nnzblocksT)
  for (n, blockT) in enumerate(nzblocksT)
    Udim = size(Us[n], 2)
    b1 = block(i1, blockT[1])
    setblock!(uind, resize(b1, Udim), n)
    Vdim = size(Vs[n], 2)
    b2 = block(i2, blockT[2])
    setblock!(vind, resize(b2, Vdim), n)
  end

  #
  # Put the blocks into U,S,V
  # 

  nzblocksU = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksS = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksV = Vector{Block{2}}(undef, nnzblocksT)

  for (n, blockT) in enumerate(nzblocksT)
    blockU = (blockT[1], UInt(n))
    nzblocksU[n] = blockU

    blockS = (n, n)
    nzblocksS[n] = blockS

    blockV = (blockT[2], UInt(n))
    nzblocksV[n] = blockV
  end

  indsU = setindex(inds(T), uind, 2)

  indsV = setindex(inds(T), vind, 1)
  indsV = permute(indsV, (2, 1))

  indsS = setindex(inds(T), dag(uind), 1)
  indsS = setindex(indsS, dag(vind), 2)

  U = BlockSparseTensor(leaf_parenttype(T), undef, nzblocksU, indsU)
  S = DiagBlockSparseTensor(
    set_eltype(leaf_parenttype(T), real(ElT)), undef, nzblocksS, indsS
  )
  V = BlockSparseTensor(leaf_parenttype(T), undef, nzblocksV, indsV)

  for n in 1:nnzblocksT
    Ub, Sb, Vb = Us[n], Ss[n], Vs[n]

    blockU = nzblocksU[n]
    blockS = nzblocksS[n]
    blockV = nzblocksV[n]

    if VERSION < v"1.5"
      # In v1.3 and v1.4 of Julia, Ub has
      # a very complicated view wrapper that
      # can't be handled efficiently
      Ub = copy(Ub)
      Vb = copy(Vb)
    end

    # <fermions>
    sU = right_arrow_sign(uind, blockU[2])

    if sU == -1
      Ub *= -1
    end
    copyto!(blockview(U, blockU), Ub)

    blockviewS = blockview(S, blockS)
    # TODO: Replace `data` with `diagview`.
    copyto!(data(blockviewS), data(Sb))

    #<fermions>
    sV = left_arrow_sign(vind, blockV[2])
    # This sign (sVP) accounts for the fact that
    # V is transposed, i.e. the index connecting to S
    # is the second index:
    sVP = 1
    if using_auto_fermion()
      sVP = -block_sign(vind, blockV[2])
    end

    if (sV * sVP) == -1
      Vb *= -1
    end
    copyto!(blockview(V, blockV), Vb)
  end
  return U, S, V, Spectrum(d, truncerr)
end

_eigen_eltypes(T::Hermitian{ElT,<:BlockSparseMatrix{ElT}}) where {ElT} = real(ElT), ElT

_eigen_eltypes(T::BlockSparseMatrix{ElT}) where {ElT} = complex(ElT), complex(ElT)

function eigen(
  T::Union{Hermitian{ElT,<:Tensor{ElT,2,<:BlockSparse}},Tensor{ElT,2,<:BlockSparse}};
  min_blockdim=nothing,
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {ElT<:Union{Real,Complex}}
  ElD, ElV = _eigen_eltypes(T)

  # Sorted eigenvalues
  d = Vector{real(ElT)}()

  for b in eachnzblock(T)
    all(==(b[1]), b) || error("Eigen currently only supports block diagonal matrices.")
  end

  b = first(eachnzblock(T))
  blockT = blockview(T, b)
  Db, Vb = eigen(blockT)
  Ds = [Db]
  Vs = [Vb]
  append!(d, abs.(data(Db)))
  for (n, b) in enumerate(eachnzblock(T))
    n == 1 && continue
    blockT = blockview(T, b)
    Db, Vb = eigen(blockT)
    push!(Ds, Db)
    push!(Vs, Vb)
    append!(d, abs.(data(Db)))
  end

  dropblocks = Int[]
  sort!(d; rev=true, by=abs)

  if any(!isnothing, (maxdim, cutoff))
    d, truncerr, docut = truncate!!(
      d; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    for n in 1:nnzblocks(T)
      blockdim = _truncated_blockdim(
        Ds[n], docut; min_blockdim, singular_values=false, truncate=true
      )
      if blockdim == 0
        push!(dropblocks, n)
      else
        # TODO: Replace call to `data` with `diagview`.
        Dtrunc = tensor(Diag(data(Ds[n])[1:blockdim]), (blockdim, blockdim))
        Ds[n] = Dtrunc
        new_size = (dim(Vs[n], 1), blockdim)
        new_data = array(Vs[n])[1:new_size[1], 1:new_size[2]]
        Vs[n] = tensor(Dense(new_data), new_size)
      end
    end
    deleteat!(Ds, dropblocks)
    deleteat!(Vs, dropblocks)
  else
    truncerr = 0.0
  end

  # Get the list of blocks of T
  # that are not dropped
  nzblocksT = nzblocks(T)
  deleteat!(nzblocksT, dropblocks)

  # The number of blocks of T remaining
  nnzblocksT = nnzblocks(T) - length(dropblocks)

  #
  # Put the blocks into D, V
  #

  i1, i2 = inds(T)
  l = sim(i1)

  lkeepblocks = Int[bT[1] for bT in nzblocksT]
  ldropblocks = setdiff(1:nblocks(l), lkeepblocks)
  deleteat!(l, ldropblocks)

  # l may have too many blocks
  (nblocks(l) > nnzblocksT) && error("New index l in eigen has too many blocks")

  # Truncation may have changed
  # some block sizes
  for n in 1:nnzblocksT
    setblockdim!(l, minimum(dims(Ds[n])), n)
  end

  r = dag(sim(l))

  indsD = (l, r)
  indsV = (dag(i2), r)

  nzblocksD = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksV = Vector{Block{2}}(undef, nnzblocksT)
  for n in 1:nnzblocksT
    blockT = nzblocksT[n]

    blockD = (n, n)
    nzblocksD[n] = blockD

    blockV = (blockT[1], n)
    nzblocksV[n] = blockV
  end

  D = DiagBlockSparseTensor(
    set_ndims(set_eltype(leaf_parenttype(T), ElD), 1), undef, nzblocksD, indsD
  )
  V = BlockSparseTensor(set_eltype(leaf_parenttype(T), ElV), undef, nzblocksV, indsV)

  for n in 1:nnzblocksT
    Db, Vb = Ds[n], Vs[n]

    blockD = nzblocksD[n]
    blockviewD = blockview(D, blockD)
    # TODO: Replace `data` with `diagview`.
    copyto!(data(blockviewD), data(Db))

    blockV = nzblocksV[n]
    copyto!(blockview(V, blockV), Vb)
  end

  return D, V, Spectrum(d, truncerr)
end

ql(T::BlockSparseTensor{<:Any,2}; kwargs...) = qx(ql, T; kwargs...)
qr(T::BlockSparseTensor{<:Any,2}; kwargs...) = qx(qr, T; kwargs...)
#
#  Generic function to implelement blocks sparse qr/ql decomposition.  It calls
#  the dense qr or ql for each block. The X tensor = R or L. 
#  This code thanks to Niklas Tausendpfund 
#  https://github.com/ntausend/variance_iTensor/blob/main/Hubig_variance_test.ipynb
#
function qx(qx::Function, T::BlockSparseTensor{<:Any,2}; positive=nothing)
  ElT = eltype(T)
  # getting total number of blocks
  nnzblocksT = nnzblocks(T)
  nzblocksT = nzblocks(T)

  Qs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)
  Xs = Vector{DenseTensor{ElT,2}}(undef, nnzblocksT)

  for (jj, b) in enumerate(eachnzblock(T))
    blockT = blockview(T, b)
    QXb = qx(blockT; positive)

    if (isnothing(QXb))
      return nothing
    end

    Q, X = QXb
    Qs[jj] = Q
    Xs[jj] = X
  end

  #
  # Make the new index connecting Q and R  
  #
  itl = ind(T, 1) #left index of T
  iq = dag(sim(itl)) #start with similar to the left index of T
  resize!(iq, nnzblocksT)  #adjust the size to match the block count
  for (n, blockT) in enumerate(nzblocksT)
    Qdim = size(Qs[n], 2) #get the block dim on right side of Q.
    b1 = block(itl, blockT[1])
    setblock!(iq, resize(b1, Qdim), n)
  end

  indsQ = setindex(inds(T), iq, 2)
  indsX = setindex(inds(T), dag(iq), 1)

  nzblocksQ = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksX = Vector{Block{2}}(undef, nnzblocksT)

  for n in 1:nnzblocksT
    blockT = nzblocksT[n]
    nzblocksQ[n] = (blockT[1], UInt(n))
    nzblocksX[n] = (UInt(n), blockT[2])
  end

  Q = BlockSparseTensor(leaf_parenttype(T), undef, nzblocksQ, indsQ)
  X = BlockSparseTensor(leaf_parenttype(T), undef, nzblocksX, indsX)

  for n in 1:nnzblocksT
    copyto!(blockview(Q, nzblocksQ[n]), Qs[n])
    copyto!(blockview(X, nzblocksX[n]), Xs[n])
  end

  Q = adapt(leaf_parenttype(T), Q)
  X = adapt(leaf_parenttype(T), X)
  return Q, X
end

function exp(
  T::Union{BlockSparseMatrix{ElT},Hermitian{ElT,<:BlockSparseMatrix{ElT}}}
) where {ElT<:Union{Real,Complex}}
  expT = BlockSparseTensor(ElT, undef, nzblocks(T), inds(T))
  for b in eachnzblock(T)
    all(==(b[1]), b) || error("exp currently supports only block-diagonal matrices")
  end
  for b in eachdiagblock(T)
    blockT = blockview(T, b)
    if isnothing(blockT)
      # Block was not found in the list, treat as 0
      id_block = Matrix{ElT}(I, blockdims(T, b))
      insertblock!(expT, b)
      blockview(expT, b) .= id_block
    else
      blockview(expT, b) .= exp(blockT)
    end
  end
  return expT
end
