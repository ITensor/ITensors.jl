using ArrayLayouts: LayoutArray
using BlockArrays: BlockArrays, BlockedUnitRange

const PermutedDimsBlockSparseArray{T,N,P,IP} = PermutedDimsArray{
  T,N,P,IP,<:AbstractBlockSparseArray{T,N}
}

function BlockArrays.blocks(a::PermutedDimsBlockSparseArray)
  return blocksparse_blocks(a)
end

function Base.similar(
  a::PermutedDimsBlockSparseArray, elt::Type, axes::Tuple{Vararg{BlockedUnitRange}}
)
  # TODO: Preserve GPU!
  return similar(BlockSparseArray{elt}, axes)
end

function Base.map!(f, a_dest::AbstractArray, a_src::PermutedDimsBlockSparseArray)
  blocksparse_map!(f, a_dest, a_src)
  return a_dest
end

# TODO: `IsWrappedBlockSparseArray` trait.
function Base.copyto!(a_dest::AbstractArray, a_src::PermutedDimsBlockSparseArray)
  map!(identity, a_dest, a_src)
  return a_dest
end

# TODO: `IsWrappedBlockSparseArray` trait.
# Fix ambiguity error
function Base.copyto!(a_dest::LayoutArray, a_src::PermutedDimsBlockSparseArray)
  map!(identity, a_dest, a_src)
  return a_dest
end
