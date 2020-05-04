
const BlockSparseMatrix{ElT,StoreT,IndsT} = BlockSparseTensor{ElT,2,StoreT,IndsT}
const DiagBlockSparseMatrix{ElT,StoreT,IndsT} = DiagBlockSparseTensor{ElT,2,StoreT,IndsT}
const DiagMatrix{ElT,StoreT,IndsT} = DiagTensor{ElT,2,StoreT,IndsT}

function _truncated_blockdim(S::DiagMatrix,
                             docut::Float64;
                             singular_values=false,
                             truncate=true)
  !truncate && return diaglength(S)
  newdim = 0
	val = singular_values ? getdiagindex(S,newdim+1)^2 : getdiagindex(S,newdim+1)
  while newdim+1 ≤ diaglength(S) && val > docut
    newdim += 1
    if newdim+1 ≤ diaglength(S)
      val = singular_values ? getdiagindex(S,newdim+1)^2 : 
             getdiagindex(S,newdim+1)
    end
  end
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
function LinearAlgebra.svd(T::BlockSparseMatrix{ElT};
                           kwargs...) where {ElT}
  alg::String = get(kwargs, :alg, "recursive")

  truncate = haskey(kwargs, :maxdim) || haskey(kwargs, :cutoff)

  Us = Vector{BlockSparseMatrix{ElT}}(undef, nnzblocks(T))
  Ss = Vector{DiagBlockSparseMatrix{real(ElT)}}(undef, nnzblocks(T))
  Vs = Vector{BlockSparseMatrix{ElT}}(undef, nnzblocks(T))

  # Sorted eigenvalues
  d = Vector{real(ElT)}()

  for n in 1:nnzblocks(T)
    b = nzblock(T, n)
    blockT = blockview(T, n)
    Ub, Sb, Vb = svd(blockT; alg = alg)
    Us[n] = Ub
    Ss[n] = Sb
    Vs[n] = Vb
    append!(d, vector(diag(Sb)))
  end

  # Square the singular values to get
  # the eigenvalues
  d .= d .^ 2
  sort!(d; rev=true)

  # Get the list of blocks of T
  # that are not dropped
  nzblocksT = nzblocks(T)

  dropblocks = Int[]
  if truncate
    truncerr,docut = truncate!(d; kwargs...)
    for n in 1:nnzblocks(T)
      blockdim = _truncated_blockdim(Ss[n],
                                     docut;
                                     singular_values=true,
                                     truncate=truncate)
      if blockdim == 0
        push!(dropblocks,n)
      else
        Strunc = tensor(Diag(store(Ss[n])[1:blockdim]),
                        (blockdim,blockdim))
        Ss[n] = Strunc
        Us[n] = copy(Us[n][1:dim(Us[n],1),1:blockdim])
        Vs[n] = copy(Vs[n][1:dim(Vs[n],1),1:blockdim])
      end
    end
    deleteat!(Ss,dropblocks)
    deleteat!(Us,dropblocks)
    deleteat!(Vs,dropblocks)
    deleteat!(nzblocksT,dropblocks)
    else
    truncerr,docut = 0.0,0.0
  end

  # The number of blocks of T remaining
  nnzblocksT = nnzblocks(T)-length(dropblocks)

  #
  # Put the blocks into U,S,V
  # 

  nb1_lt_nb2 = (nblocks(T)[1] < nblocks(T)[2] || 
                (nblocks(T)[1] == nblocks(T)[2] && 
                 dim(T,1) < dim(T,2)))

  if nb1_lt_nb2
    uind = sim(ind(T,1))
  else
    uind = sim(ind(T,2))
  end

  deleteat!(uind,dropblocks)

  # uind may have too many blocks
  if nblocks(uind) > nnzblocksT
    resize!(uind,nnzblocksT)
  end

  for n in 1:nnzblocksT
    setblockdim!(uind, minimum(dims(Ss[n])), n)
  end

  if dir(uind) != dir(inds(T)[1])
    uind = dag(uind)
  end
  indsU = setindex(inds(T), dag(uind), 2)

  vind = sim(uind)
  if dir(vind) != dir(inds(T)[2])
    vind = dag(vind)
  end
  indsV = setindex(inds(T),dag(vind),1)
  indsV = permute(indsV,(2,1))

  indsS = setindex(inds(T),uind,1)
  indsS = setindex(indsS,vind,2)

  nzblocksU = Vector{Block{2}}(undef,nnzblocksT)
  nzblocksS = Vector{Block{2}}(undef,nnzblocksT)
  nzblocksV = Vector{Block{2}}(undef,nnzblocksT)

  for n in 1:nnzblocksT
    blockT = nzblocksT[n]

    blockU = (blockT[1],n)
    nzblocksU[n] = blockU

    blockS = (n,n)
    nzblocksS[n] = blockS

    blockV = (blockT[2],n)
    nzblocksV[n] = blockV
  end

  U = BlockSparseTensor(undef, nzblocksU, indsU)
  V = BlockSparseTensor(undef, nzblocksV, indsV)
  S = DiagBlockSparseTensor(undef, nzblocksS, indsS)

  for n in 1:nnzblocksT
    Ub, Sb, Vb = Us[n], Ss[n], Vs[n]

    blockU = nzblocksU[n]
    blockV = nzblocksV[n]
    blockS = nzblocksS[n]

    blockview(U, blockU) .= Ub
    blockview(V, blockV) .= Vb

    blockviewS = blockview(S, blockS)
    for i in 1:diaglength(Sb)
      setdiagindex!(blockviewS, getdiagindex(Sb, i), i)
    end
  end

  return U,S,V,Spectrum(d,truncerr)
end

_eigen_eltypes(T::Hermitian{ElT,<:BlockSparseMatrix{ElT}}) where {ElT} = real(ElT), ElT

_eigen_eltypes(T::BlockSparseMatrix{ElT}) where {ElT} = complex(ElT), complex(ElT)

function LinearAlgebra.eigen(T::Union{Hermitian{ElT,<:BlockSparseMatrix{ElT}},
                                      BlockSparseMatrix{ElT}};
                             kwargs...) where {ElT<:Union{Real,Complex}}
  truncate = haskey(kwargs,:maxdim) || haskey(kwargs,:cutoff)

  ElD, ElV = _eigen_eltypes(T)

  # Sorted eigenvalues
  d = Vector{real(ElT)}()

  b = nzblock(T, 1)
  all(==(b[1]), b) || error("Eigen currently only supports block diagonal matrices.")
  blockT = blockview(T, 1)
  Db, Vb = eigen(blockT)
  Ds = [Db]
  Vs = [Vb]
  append!(d, abs.(vector(diag(Db))))
  for n in 2:nnzblocks(T)
    b = nzblock(T, n)
    all(==(b[1]), b) || error("Eigen currently only supports block diagonal matrices.")
    blockT = blockview(T, n)
    Db, Vb = eigen(blockT)
    push!(Ds, Db)
    push!(Vs, Vb)
    append!(d, abs.(vector(diag(Db))))
  end

  dropblocks = Int[]
  sort!(d; rev=true, by=abs)

  if truncate
    truncerr,docut = truncate!(d; kwargs...)
    for n in 1:nnzblocks(T)
      blockdim = _truncated_blockdim(Ds[n],docut)
      if blockdim == 0
        push!(dropblocks,n)
      else
        Dtrunc = tensor(Diag(store(Ds[n])[1:blockdim]),
                        (blockdim,blockdim))
        Ds[n] = Dtrunc
        Vs[n] = copy(Vs[n][1:dim(Vs[n],1),1:blockdim])
      end
    end
    deleteat!(Ds,dropblocks)
    deleteat!(Vs,dropblocks)
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

  deleteat!(l, dropblocks)

  # l may have too many blocks
  if nblocks(l) > nnzblocksT
    resize!(l, nnzblocksT)
  end

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

  D = DiagBlockSparseTensor(ElD, undef, nzblocksD, indsD)
  V = BlockSparseTensor(ElV, undef, nzblocksV, indsV)

  for n in 1:nnzblocksT
    Db, Vb = Ds[n], Vs[n]

    blockD = nzblocksD[n]
    blockviewD = blockview(D,blockD)
    for i in 1:diaglength(Db)
      setdiagindex!(blockviewD, getdiagindex(Db, i), i)
    end

    blockV = nzblocksV[n]
    blockview(V, blockV) .= Vb
  end

  return D, V, Spectrum(d, truncerr)
end

