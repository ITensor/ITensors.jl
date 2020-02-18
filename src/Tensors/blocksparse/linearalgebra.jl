
const BlockSparseMatrix{ElT,StoreT,IndsT} = BlockSparseTensor{ElT,2,StoreT,IndsT}

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
  nb1_lt_nb2 = (nblocks(T)[1] < nblocks(T)[2] || (nblocks(T)[1] == nblocks(T)[2] && dim(T,1) < dim(T,2)))

  if nb1_lt_nb2
    uind = sim(ind(T,1))
    nzblocksT = nzblocks(T)
    for n in 1:nblocks(uind)
      b = findfirst(i->i[1]==n,nzblocksT)
      if !isnothing(b)
        blockT = nzblocksT[b]
        #setindex!(uind,minimum(blockdims(T,blockT)),n)
        setblockdim!(uind,minimum(blockdims(T,blockT)),n)
      end
    end
  else
    uind = sim(ind(T,2))
    nzblocksT = nzblocks(T)
    for n in 1:nblocks(uind)
      b = findfirst(i->i[2]==n,nzblocksT)
      if !isnothing(b)
        blockT = nzblocksT[b]
        #setindex!(uind,minimum(blockdims(T,blockT)),n)
        setblockdim!(uind,minimum(blockdims(T,blockT)),n)
      end
    end
  end

  indsU = setindex(inds(T),uind,2)

  if nb1_lt_nb2
    # Make U block diagonal by convention
    blocksU = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsU))]
    U = BlockSparseTensor(undef,blocksU,indsU)
  else
    U = BlockSparseTensor(ElT,undef,blockoffsets(T),indsU)
  end
  
  vind = sim(uind)
  indsV = setindex(inds(T),vind,1)

  if nb1_lt_nb2
    blockoffsetsV,indsV = permutedims(blockoffsets(T),indsV,(2,1))
    V = BlockSparseTensor(ElT,undef,blockoffsetsV,indsV)
  else
    blocksV = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsV))]
    V = BlockSparseTensor(undef,blocksV,indsV)
  end

  indsS = setindex(inds(T),uind,1)
  indsS = setindex(indsS,vind,2)

  # Make S block diagonal by convention
  blocksS = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsS))]

  # TODO: make a DiagBlockTensor type
  # S = DiagBlockTensor(blocksS,indsS)
  S = BlockSparseTensor(blocksS,indsS)

  for n in 1:nnzblocks(T)
    b = block(T,n)
    blockT = blockview(T,n)
    Ub,Sb,Vb = svd(blockT)
    if nb1_lt_nb2
      # Block of V, permute since we
      # are returning V such that T = U*S*V'
      bV = permute(b,(2,1))

      blockview(V,bV) .= Vb

      blockview(U,(bV[2],bV[2])) .= Ub
      Sblock = blockview(S,(bV[2],bV[2]))
      for i in 1:diag_length(Sb)
        Sblock[i,i] = getdiag(Sb,i)
      end
    else
      blockview(U,b) .= Ub
      blockview(V,n) .= Vb
      for i in 1:diag_length(Sb)
        blockview(S,n)[i,i] = getdiag(Sb,i)
      end
    end
  end
  # TODO: output spec
  return U,S,V,Spectrum(Float64[],0.0)
end

