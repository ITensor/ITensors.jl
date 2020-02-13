
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
  dim(T,1) < dim(T,2) && error("svd(::BlockSparseTensor) only implemented for dim(T,1) >= dim(T,2)")

  indsU = inds(T)
  uind = sim(ind(T,1))
  indsU = setindex(indsU,uind,2)

  # Make U block diagonal by convention
  blocksU = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsU))]

  U = BlockSparseTensor(undef,blocksU,indsU)
  
  indsV = inds(T)
  vind = sim(ind(T,1))
  indsV = setindex(indsV,vind,1)

  V = BlockSparseTensor(ElT,undef,blockoffsets(T),indsV)

  indsS = inds(T)
  indsS = setindex(indsS,uind,1)
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
    # Block of V, permute since we
    # are returning V such that T = U*S*V'
    bV = permute(b,(2,1))
    blockview(V,bV) .= Vb
    blockview(U,(bV[2],bV[2])) .= Ub
    Sblock = blockview(S,(bV[2],bV[2]))
    for i in 1:diag_length(Sb)
      Sblock[i,i] = getdiag(Sb,i)
    end
  end
  return U,S,V
end

