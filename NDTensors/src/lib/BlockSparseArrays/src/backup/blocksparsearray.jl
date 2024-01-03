using BlockArrays: block
## using ..SparseArrayDOKs: SparseArrayDOK

# Also add a version with contiguous underlying data.
struct BlockSparseArray{
  T,
  N,
  A<:AbstractArray{T,N},
  Blocks<:AbstractArray{A,N},
  Axes<:NTuple{N,AbstractUnitRange{Int}},
} <: AbstractBlockSparseArray{T,N}
  blocks::Blocks
  axes::Axes
end

Base.axes(block_arr::BlockSparseArray) = block_arr.axes
BlockArrays.blocks(a::BlockSparseArray) = a.blocks
# TODO: Use `SetParameters`.
blocktype(a::BlockSparseArray{<:Any,<:Any,A}) where {A} = A

# TODO: Use `SetParameters`.
set_ndims(::Type{<:Array{T}}, n) where {T} = Array{T,n}

# TODO: Move to `AbstractArray` file.
function nonzero_blockkeys(a::AbstractArray)
  return map(Block ∘ Tuple, collect(nonzero_keys(blocks(a))))
end

function Base.reshape(a::BlockSparseArray, ax::Tuple{Vararg{AbstractUnitRange}})
  ## TODO: Use `SparseArray` reshape in some way?
  ## blocks_reshaped = reshape(blocks(a), blocklength.(ax))
  blocktype_reshaped = set_ndims(blocktype(a), length(ax))
  # TODO: Some other way of getting `zero` function?
  blocks_reshaped = default_sparsearray_type()(
    Dictionary{CartesianIndex{length(ax)},blocktype_reshaped}(),
    blocklength.(ax),
    BlockZero(ax),
  )
  for I in nonzero_keys(blocks(a))
    i = LinearIndices(blocks(a))[I]
    I_reshaped = CartesianIndices(blocks_reshaped)[i]
    blocks_reshaped[I_reshaped] = reshape(
      a[Block(Tuple(I))], block_size(ax, Block(Tuple(I_reshaped)))
    )
  end
  a_reshaped = BlockSparseArray(blocks_reshaped, ax)
  return a_reshaped
end

function Base.reshape(a::BlockSparseArray, blockinds::Tuple{Vararg{AbstractVector}})
  return reshape(a, blockedrange.(blockinds))
end

# The size of a block
function block_size(axes::Tuple{Vararg{AbstractUnitRange}}, block::Block)
  return length.(getindex.(axes, blocktuple(block)))
end

# The size of a block
function block_size(blockinds::Tuple{Vararg{AbstractVector}}, block::Block)
  return block_size(blockedrange.(blockinds), block)
end

struct BlockZero{Axes}
  axes::Axes
end

function (f::BlockZero)(a::AbstractArray, I::CartesianIndex)
  return f(eltype(a), I)
end

function (f::BlockZero)(
  arraytype::Type{<:AbstractArray{<:Any,N}}, I::CartesianIndex{N}
) where {N}
  # TODO: Make sure this works for sparse or block sparse blocks, immutable
  # blocks, diagonal blocks, etc.!
  return fill!(arraytype(undef, block_size(f.axes, Block(Tuple(I)))), false)
end

# Fallback so that `SparseArray` with scalar elements works.
function (f::BlockZero)(blocktype::Type{<:Number}, I::CartesianIndex)
  return zero(blocktype)
end

# Fallback to Array if it is abstract
function (f::BlockZero)(
  arraytype::Type{AbstractArray{T,N}}, I::CartesianIndex{N}
) where {T,N}
  return fill!(Array{T,N}(undef, block_size(f.axes, Block(Tuple(I)))), false)
end

function BlockSparseArray(
  blocks::AbstractVector{<:Block{N}}, blockdata::AbstractVector, axes::Tuple{Vararg{Any,N}}
) where {N}
  return BlockSparseArray(Dictionary(blocks, blockdata), axes)
end

function BlockSparseArray{T,N,B}(
  ::UndefInitializer, blocks::AbstractVector{<:Block{N}}, axes::Tuple{Vararg{Any,N}}
) where {T,N,B}
  blockdata = map(block -> B(undef, block_size(axes, block)), Indices(blocks))
  return BlockSparseArray(blockdata, axes)
end

function BlockSparseArray{T,N}(::UndefInitializer, axes::Tuple{Vararg{Any,N}}) where {T,N}
  B = Array{T,N} # TODO: Use `default_blocktype`.
  return BlockSparseArray{T,N,B}(undef, axes)
end

function BlockSparseArray{T,N,B}(
  ::UndefInitializer, axes::Tuple{Vararg{Any,N}}
) where {T,N,B}
  # TODO: `Block{N,Int}`?
  blocks = Vector{Block{N}}()
  return BlockSparseArray{T,N,B}(undef, blocks, axes)
end

## struct BlockSparseArray{
##   T,N,A<:AbstractArray{T,N},Blocks<:SparseArray{A,N},Axes<:NTuple{N,AbstractUnitRange{Int}}
## } <: AbstractBlockArray{T,N}
##   blocks::Blocks
##   axes::Axes
## end

function BlockSparseArray(a::AbstractArray, axes::Tuple{Vararg{AbstractUnitRange}})
  A = eltype(a)
  T = eltype(A)
  N = ndims(a)
  Blocks = typeof(a)
  Axes = typeof(axes)
  return BlockSparseArray{T,N,A,Blocks,Axes}(a, axes)
end

function BlockSparseArray(
  blockdata::Dictionary{<:Block{N}}, axes::Tuple{Vararg{AbstractUnitRange{Int},N}}
) where {N}
  blocks = keys(blockdata)
  cartesianblocks = if isempty(blockdata)
    Dictionary{Block{N},CartesianIndex{N}}()
  else
    map(block -> CartesianIndex(inttuple(block)), blocks)
  end
  cartesiandata = Dictionary(cartesianblocks, blockdata)
  block_storage = default_sparsearray_type()(
    cartesiandata, blocklength.(axes), BlockZero(axes)
  )
  return BlockSparseArray(block_storage, axes)
end

function BlockSparseArray(
  blockdata::Dictionary{<:Block{N}}, blockinds::Tuple{Vararg{AbstractVector,N}}
) where {N}
  return BlockSparseArray(blockdata, blockedrange.(blockinds))
end

# Empty constructors
function BlockSparseArray{T,N,A}(
  blockinds::Tuple{Vararg{AbstractVector,N}}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray(Dictionary{Block{N},A}(), blockinds)
end

function BlockSparseArray{T,N,A}(
  blockinds::Vararg{AbstractVector,N}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(blockinds)
end

function BlockSparseArray{T,N}(blockinds::Tuple{Vararg{AbstractVector,N}}) where {T,N}
  # TODO: Use default function.
  return BlockSparseArray{T,N,Array{T,N}}(blockinds)
end

function BlockSparseArray{T,N}(blockinds::Vararg{AbstractVector,N}) where {T,N}
  # TODO: Use default function.
  return BlockSparseArray{T,N,Array{T,N}}(blockinds)
end

function BlockSparseArray{T}(blockinds::Tuple{Vararg{AbstractVector,N}}) where {T,N}
  # TODO: Use default function.
  return BlockSparseArray{T,N,Array{T,N}}(blockinds)
end

function BlockSparseArray{T}(blockinds::Vararg{AbstractVector,N}) where {T,N}
  # TODO: Use default function.
  return BlockSparseArray{T,N,Array{T,N}}(blockinds)
end

function Base.copy(block_arr::BlockSparseArray)
  return BlockSparseArray(deepcopy(block_arr.blocks), copy.(block_arr.axes))
end

function BlockArrays.viewblock(block_arr::BlockSparseArray, block)
  blks = inttuple(block)
  @boundscheck blockcheckbounds(block_arr, blks...)
  ## block_size = length.(getindex.(axes(block_arr), Block.(blks)))
  # TODO: Make this `Zeros`?
  ## zero = zeros(eltype(block_arr), block_size)
  return block_arr.blocks[blks...] # Fails because zero isn't defined
end

function _getindex(block_arr::BlockSparseArray{T,N}, bi::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(block_arr, Block(bi.I))
  bl = view(block_arr, block(bi))
  inds = bi.α
  @boundscheck checkbounds(bl, inds...)
  v = bl[inds...]
  return v
end

function Base.getindex(block_arr::BlockSparseArray{T,N}, bi::BlockIndex{N}) where {T,N}
  return _getindex(block_arr, bi)
end

# Avoid ambiguity error
function Base.getindex(block_arr::BlockSparseArray{T,1}, bi::BlockIndex{1}) where {T}
  return _getindex(block_arr, bi)
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
  block_arr::BlockSparseArray{T,N}, v, block::Tuple{Vararg{Integer,N}}
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

# Fixes ambiguity error.
# TODO: Is this needed?
function Base.getindex(block_arr::BlockSparseArray{<:Any,0})
  return blocks(block_arr)[CartesianIndex()][]
end

function Base.permutedims!(a_src::BlockSparseArray, a_dest::BlockSparseArray, perm)
  copyto!(a_src, PermutedDimsArray(a_dest, perm))
  return a_src
end

# TODO: Call `map(b -> permutedims(b, perm), blocks(a))`
# or something like that.
function Base.permutedims(a::BlockSparseArray, perm)
  a_dest = zero(PermutedDimsArray(a, perm))
  permutedims!(a_dest, a, perm)
  return a_dest
end

# TODO: Make `PermutedBlockSparseArray`.
function BlockArrays.blocks(
  a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:BlockSparseArray}
)
  return PermutedDimsArray(blocks(parent(a)), perm(a))
end

# TODO: Make `PermutedBlockSparseArray`.
function BlockArrays.blocks(a::Hermitian{<:Any,<:BlockSparseArray})
  return Hermitian(blocks(parent(a)))
end

# TODO: Make `PermutedBlockSparseArray`.
function Base.zero(a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:BlockSparseArray})
  return BlockSparseArray(zero(blocks(a)), axes(a))
end

# TODO: Make `PermutedBlockSparseArray`.
function Base.copyto!(
  a_src::BlockSparseArray,
  a_dest::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:BlockSparseArray},
)
  map_nonzeros!(x -> permutedims(x, perm(a_dest)), blocks(a_src), blocks(a_dest))
  return a_src
end

function Base.map(f, as::BlockSparseArray...)
  @assert allequal(axes.(as))
  return BlockSparseArray(map(f, blocks.(as)...), axes(first(as)))
end

function Base.map!(f, a_dest::BlockSparseArray, as::BlockSparseArray...)
  @assert allequal(axes.((a_dest, as...)))
  return BlockSparseArray(map!(f, blocks(a_dest), blocks.(as)...), axes(a_dest))
end
