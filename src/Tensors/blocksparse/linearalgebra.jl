
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

  @show inds(T)

  indsU = inds(T)
  uind = sim(ind(T,1))
  indsU = setindex(indsU,uind,2)

  @show indsU
  @show nblocks(indsU)

  # Make U block diagonal by convention
  blocksU = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsU))]

  @show blocksU

  U = BlockSparseTensor(undef,blocksU,indsU)
  
  @show U

  indsV = inds(T)
  vind = sim(ind(T,1))
  indsV = setindex(indsV,vind,1)

  @show indsV

  V = BlockSparseTensor(ElT,undef,blockoffsets(T),indsV)

  @show V

  indsS = inds(T)
  indsS = setindex(indsS,uind,1)
  indsS = setindex(indsS,vind,2)

  @show indsS

  # Make S block diagonal by convention
  blocksS = Block{2}[ntuple(_->i,Val(2)) for i = 1:minimum(nblocks(indsS))]

  @show blocksS

  # TODO: make a DiagBlockTensor type
  # S = DiagBlockTensor(blocksS,indsS)
  S = BlockSparseTensor(blocksS,indsS)

  @show S

  @show nzblocks(T)
  for n in 1:nnzblocks(T)
    @show n

    b = block(T,n)

    @show b

    blockT = blockview(T,n)

    @show blockT

    Ub,Sb,Vb = svd(blockT)

    @show Ub,Sb,Vb

    blockview(U,n) .= Ub
    blockview(V,n) .= Vb'

    @show typeof(Sb)

    for i in 1:diag_length(Sb)
      blockview(S,n)[i,i] = getdiag(Sb,i)
    end
  end

  @show T
  @show U
  @show S
  @show V

  return U,S,V
end

