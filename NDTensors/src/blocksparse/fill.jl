#
# Random
#

function randn(
  StorageT::Type{<:BlockSparse{ElT}}, blockoffsets::BlockOffsets, dim::Integer
) where {ElT<:Number}
  return randn(Random.default_rng(), StorageT, blockoffsets, dim)
end

function randn(
  rng::AbstractRNG, ::Type{<:BlockSparse{ElT}}, blockoffsets::BlockOffsets, dim::Integer
) where {ElT<:Number}
  return BlockSparse(randn(rng, ElT, dim), blockoffsets)
end

function randn(
  TensorT::Type{<:BlockSparseTensor{ElT,N}}, blocks::Vector{<:BlockT}, inds
) where {ElT,BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  return randn(Random.default_rng(), TensorT, blocks, inds)
end

function randn(
  rng::AbstractRNG, ::Type{<:BlockSparseTensor{ElT,N}}, blocks::Vector{<:BlockT}, inds
) where {ElT,BlockT<:Union{Block{N},NTuple{N,<:Integer}}} where {N}
  boffs, nnz = blockoffsets(blocks, inds)
  storage = randn(rng, BlockSparse{ElT}, boffs, nnz)
  return tensor(storage, inds)
end


#
# Zero
#

function zeros(
  ::BlockSparseTensor{ElT,N}, blockoffsets::BlockOffsets{N}, inds
) where {ElT,N}
  return BlockSparseTensor(ElT, blockoffsets, inds)
end

function zeros(
  ::Type{<:BlockSparseTensor{ElT,N}}, blockoffsets::BlockOffsets{N}, inds
) where {ElT,N}
  return BlockSparseTensor(ElT, blockoffsets, inds)
end

function zeros(tensortype::Type{<:BlockSparseTensor}, inds)
  return BlockSparseTensor(datatype(tensortype), inds)
end

zeros(tensor::BlockSparseTensor, inds) = zeros(typeof(tensor), inds)