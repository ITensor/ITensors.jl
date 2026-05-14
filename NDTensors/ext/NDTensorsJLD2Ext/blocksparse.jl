using JLD2: JLD2
using NDTensors: NDTensors, BlockSparse, SerializedBlockSparse, _deserialize_blockoffsets,
    _serialize_blockoffsets

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
JLD2.rconvert(::Type{S}, bs::S) where {T, S <: BlockSparse{T}} = bs

# Uses unparameterized BlockSparse constructor because S(...) doesn't exist in NDTensors.
function JLD2.rconvert(::Type{S}, s) where {T, S <: BlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return BlockSparse(s.data, _deserialize_blockoffsets(s.ndims, s))::S
end
