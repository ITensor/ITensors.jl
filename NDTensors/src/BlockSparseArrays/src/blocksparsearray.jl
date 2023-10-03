using BlockArrays: block

# Also add a version with contiguous underlying data.
struct BlockSparseArray{
  T,N,Blocks<:SparseArray{<:AbstractArray{T,N},N},Axes<:NTuple{N,AbstractUnitRange{Int}}
} <: AbstractBlockArray{T,N}
  blocks::Blocks
  axes::Axes
end

# The size of a block
function block_size(axes::Tuple, block::Block)
  return length.(getindex.(axes, Block.(block.n)))
end

struct BlockZero{Axes}
  axes::Axes
end

function (f::BlockZero)(T::Type, I::CartesianIndex)
  return fill!(T(undef, block_size(f.axes, Block(Tuple(I)))), false)
end

function BlockSparseArray(
  blocks::AbstractVector{<:Block{N}}, blockdata::AbstractVector, axes::NTuple{N}
) where {N}
  return BlockSparseArray(Dictionary(blocks, blockdata), axes)
end

function BlockSparseArray(
  blockdata::Dictionary{<:Block{N}}, axes::NTuple{N,AbstractUnitRange{Int}}
) where {N}
  blocks = keys(blockdata)
  cartesianblocks = map(block -> CartesianIndex(block.n), blocks)
  cartesiandata = Dictionary(cartesianblocks, blockdata)
  block_storage = SparseArray(cartesiandata, blocklength.(axes), BlockZero(axes))
  return BlockSparseArray(block_storage, axes)
end

function BlockSparseArray(
  blockdata::Dictionary{<:Block{N}}, blockinds::NTuple{N,AbstractVector}
) where {N}
  return BlockSparseArray(blockdata, blockedrange.(blockinds))
end

Base.axes(block_arr::BlockSparseArray) = block_arr.axes

function Base.copy(block_arr::BlockSparseArray)
  return BlockSparseArray(deepcopy(block_arr.blocks), copy.(block_arr.axes))
end

function BlockArrays.viewblock(block_arr::BlockSparseArray, block)
  blks = block.n
  @boundscheck blockcheckbounds(block_arr, blks...)
  ## block_size = length.(getindex.(axes(block_arr), Block.(blks)))
  # TODO: Make this `Zeros`?
  ## zero = zeros(eltype(block_arr), block_size)
  return block_arr.blocks[blks...] # Fails because zero isn't defined
  ## return get_nonzero(block_arr.blocks, blks, zero)
end

function Base.getindex(block_arr::BlockSparseArray{T,N}, bi::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(block_arr, Block(bi.I))
  bl = view(block_arr, block(bi))
  inds = bi.α
  @boundscheck checkbounds(bl, inds...)
  v = bl[inds...]
  return v
end

function Base.setindex!(
  block_arr::BlockSparseArray{T,N}, v, i::Vararg{Integer,N}
) where {T,N}
  @boundscheck checkbounds(block_arr, i...)
  block_indices = findblockindex.(axes(block_arr), i)
  block = map(block_index -> Block(block_index.I), block_indices)
  offsets = map(block_index -> only(block_index.α), block_indices)
  block_view = @view block_arr[block...]
  block_view[offsets...] = v
  block_arr[block...] = block_view
  return block_arr
end

function BlockArrays._check_setblock!(
  block_arr::BlockSparseArray{T,N}, v, block::NTuple{N,Integer}
) where {T,N}
  for i in 1:N
    bsz = length(axes(block_arr, i)[Block(block[i])])
    if size(v, i) != bsz
      throw(
        DimensionMismatch(
          string(
            "tried to assign $(size(v)) array to ",
            length.(getindex.(axes(block_arr), block)),
            " block",
          ),
        ),
      )
    end
  end
end

function Base.setindex!(
  block_arr::BlockSparseArray{T,N}, v, block::Vararg{Block{1},N}
) where {T,N}
  blks = Int.(block)
  @boundscheck blockcheckbounds(block_arr, blks...)
  @boundscheck BlockArrays._check_setblock!(block_arr, v, blks)
  # This fails since it tries to replace the element
  block_arr.blocks[blks...] = v
  # Use .= here to overwrite data.
  ## block_view = @view block_arr[Block(blks)]
  ## block_view .= v
  return block_arr
end

function Base.getindex(block_arr::BlockSparseArray{T,N}, i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(block_arr, i...)
  v = block_arr[findblockindex.(axes(block_arr), i)...]
  return v
end
