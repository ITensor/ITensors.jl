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
