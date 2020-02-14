
"""
svd(T::BlockSparseTensor{<:Number,2}; kwargs...)

svd of an order-2 BlockSparseTensor.

This function assumes that there is one block
per row/column, otherwise it fails.
This assumption makes it so the result can be
computed from the dense svds of seperate blocks.
"""
function LinearAlgebra.svd(T::BlockSparseTensor{ElT,2,IndsT};
                           kwargs...) where {ElT,IndsT}
  nb1_lt_nb2 = (nblocks(T)[1] < nblocks(T)[2] || (nblocks(T)[1] == nblocks(T)[2] && dim(T,1) < dim(T,2)))

  if nb1_lt_nb2
    uind = sim(ind(T,1))
  else
    uind = sim(ind(T,2))
  end

  indsU = setindex(inds(T),uind,2)

  if nb1_lt_nb2
    # Make U block diagonal by convention
    blocksU = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsU))]
    U = BlockSparseTensor(undef,blocksU,indsU)
  else
    U = BlockSparseTensor(ElT,undef,blockoffsets(T),indsU)
  end
  
  if nb1_lt_nb2
    vind = sim(ind(T,1))
  else
    vind = sim(ind(T,2))
  end
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
      @show dims(Ub)
      @show dims(blockview(U,b))
      blockview(U,b) .= Ub
      blockview(V,n) .= Vb
      for i in 1:diag_length(Sb)
        blockview(S,n)[i,i] = getdiag(Sb,i)
      end
    end
  end
  return U,S,V
end

