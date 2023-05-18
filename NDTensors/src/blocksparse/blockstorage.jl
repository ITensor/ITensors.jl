#
# Block storage type
#

struct BlockStorage{ElT, VecT, N} <: TensorStorage{ElT}
  blockoffsets::BlockOffsets{N}
  storagesets::Vector{TensorStorage}
  data::VecT
  function BlockStorage(
    data::VecT, storagesets, blockoffsets::BlockOffsets{N}
  ) where {VecT<:AbstractVector{ElT},N} where {ElT}
    return new{ElT,VecT,N}(data, blockoffsets)
  end
end

