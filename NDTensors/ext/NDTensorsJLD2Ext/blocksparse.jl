using JLD2: JLD2
using NDTensors: NDTensors, Block, BlockOffsets, BlockSparse, SerializedBlockSparse

# Shared block-offset (de)serialization helpers, used here for BlockSparse and reused
# from diagblocksparse.jl for DiagBlockSparse.
function _serialize_blockoffsets(storage)
    boffs = NDTensors.blockoffsets(storage)
    block_indices = Vector{Int}[collect(Int, Tuple(block)) for block in keys(boffs)]
    block_offsets = collect(Int, values(boffs))
    return (; block_indices, block_offsets)
end

function _deserialize_blockoffsets(ndims, s)
    boffs = BlockOffsets{ndims}()
    for (block_idx, offset) in zip(s.block_indices, s.block_offsets)
        insert!(boffs, Block(NTuple{ndims, UInt}(block_idx)), offset)
    end
    return boffs
end

JLD2.writeas(::Type{<:BlockSparse{T}}) where {T} = SerializedBlockSparse{T}

function JLD2.wconvert(
        ::Type{SerializedBlockSparse{T}}, bs::BlockSparse{T, <:Any, N}
    ) where {T, N}
    version = 1
    (; block_indices, block_offsets) = _serialize_blockoffsets(bs)
    return SerializedBlockSparse{T}(
        version, N, convert(Vector{T}, NDTensors.data(bs)), block_indices, block_offsets
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
    return BlockSparse(s.data, _deserialize_blockoffsets(s.ndims, s))::S
end
