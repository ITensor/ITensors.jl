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
