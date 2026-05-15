using JLD2: JLD2
using NDTensors: NDTensors, Block, BlockOffsets, BlockSparse, SerializedBlockSparse

# Shared block-offset (de)serialization helpers, used here for BlockSparse and reused
# from diagblocksparse.jl for DiagBlockSparse. Block positions are written as the
# columns of a `(ndims, num_blocks)` `Matrix{Int64}` (COO sparse-tensor convention).

function _serialize_blockoffsets(::Val{N}, storage) where {N}
    boffs = NDTensors.blockoffsets(storage)
    nblocks = length(boffs)
    block_indices = Matrix{Int64}(undef, N, nblocks)
    block_offsets = Vector{Int64}(undef, nblocks)
    for (i, (block, offset)) in enumerate(pairs(boffs))
        block_indices[:, i] .= Tuple(block)
        block_offsets[i] = offset
    end
    return (; block_indices, block_offsets)
end

function _deserialize_blockoffsets(::Val{N}, s) where {N}
    boffs = BlockOffsets{N}()
    for j in 1:size(s.block_indices, 2)
        block_tuple = NTuple{N, UInt}(@view s.block_indices[:, j])
        insert!(boffs, Block(block_tuple), s.block_offsets[j])
    end
    return boffs
end

JLD2.writeas(::Type{<:BlockSparse{T}}) where {T} = SerializedBlockSparse{T}

function JLD2.wconvert(
        ::Type{SerializedBlockSparse{T}}, bs::BlockSparse{T, <:Any, N}
    ) where {T, N}
    version = UInt32(1)
    (; block_indices, block_offsets) = _serialize_blockoffsets(Val(N), bs)
    return SerializedBlockSparse{T}(
        version, convert(Vector{T}, NDTensors.data(bs)), block_indices, block_offsets
    )
end

# Workaround for a JLD2 bug where rconvert is called twice for types with custom
# serialization that appear as fields inside other compound types.
# TODO: Remove this idempotent method once the JLD2 bug is fixed.
JLD2.rconvert(::Type{S}, bs::S) where {T, S <: BlockSparse{T}} = bs

# Uses unparameterized BlockSparse constructor because S(...) doesn't exist in NDTensors.
function JLD2.rconvert(::Type{S}, s) where {T, S <: BlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    N = size(s.block_indices, 1)
    return BlockSparse(s.data, _deserialize_blockoffsets(Val(N), s))::S
end
