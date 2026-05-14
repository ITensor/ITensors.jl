# On-disk struct layouts for the NDTensors storage types. Used by NDTensorsJLD2Ext
# (and intended to be reusable by other serialization backends in the future). Kept in
# the main package so that the type names recorded in serialized files do not encode the
# extension module's namespace.

struct SerializedDense{T}
    version::Int
    data::Vector{T}
end

struct SerializedBlockSparse{T}
    version::Int
    ndims::Int
    data::Vector{T}
    block_indices::Vector{Vector{Int}}
    block_offsets::Vector{Int}
end

struct SerializedDiag{T}
    version::Int
    data::Vector{T}
end

struct SerializedUniformDiag{T}
    version::Int
    value::T
end

struct SerializedDiagBlockSparse{T}
    version::Int
    ndims::Int
    data::Vector{T}
    block_indices::Vector{Vector{Int}}
    block_offsets::Vector{Int}
end

struct SerializedUniformDiagBlockSparse{T}
    version::Int
    ndims::Int
    value::T
    block_indices::Vector{Vector{Int}}
    block_offsets::Vector{Int}
end

struct SerializedEmptyStorage{T}
    version::Int
    eltype::Type{T}
end

# Shared block-offset (de)serialization helpers, used by BlockSparse and DiagBlockSparse.
function _serialize_blockoffsets(storage)
    boffs = blockoffsets(storage)
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
