using JLD2: JLD2
using NDTensors: NDTensors, DiagBlockSparse, NonuniformDiagBlockSparse,
    SerializedDiagBlockSparse, SerializedUniformDiagBlockSparse, UniformDiagBlockSparse

# `_serialize_blockoffsets` / `_deserialize_blockoffsets` are defined in blocksparse.jl
# and visible here via the shared NDTensorsJLD2Ext module scope.

# --- DiagBlockSparse (nonuniform) ---

function JLD2.writeas(::Type{<:NonuniformDiagBlockSparse{T}}) where {T}
    return SerializedDiagBlockSparse{T}
end

function JLD2.wconvert(
        ::Type{SerializedDiagBlockSparse{T}}, dbs::DiagBlockSparse{T, <:AbstractVector, N}
    ) where {T, N}
    version = 1
    (; block_indices, block_offsets) = _serialize_blockoffsets(dbs)
    return SerializedDiagBlockSparse{T}(
        version, N, convert(Vector{T}, NDTensors.data(dbs)), block_indices, block_offsets
    )
end

JLD2.rconvert(::Type{S}, d::S) where {T, S <: NonuniformDiagBlockSparse{T}} = d

# Uses unparameterized DiagBlockSparse constructor because S(...) doesn't exist in NDTensors.
function JLD2.rconvert(::Type{S}, s) where {T, S <: NonuniformDiagBlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return DiagBlockSparse(s.data, _deserialize_blockoffsets(s.ndims, s))::S
end

# --- DiagBlockSparse (uniform) ---

function JLD2.writeas(::Type{<:UniformDiagBlockSparse{T}}) where {T}
    return SerializedUniformDiagBlockSparse{T}
end

function JLD2.wconvert(
        ::Type{SerializedUniformDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:Number, N}
    ) where {T, N}
    version = 1
    (; block_indices, block_offsets) = _serialize_blockoffsets(dbs)
    return SerializedUniformDiagBlockSparse{T}(
        version, N, convert(T, NDTensors.data(dbs)), block_indices, block_offsets
    )
end

JLD2.rconvert(::Type{S}, d::S) where {T, S <: UniformDiagBlockSparse{T}} = d

# Uses unparameterized DiagBlockSparse constructor because S(...) doesn't exist in NDTensors.
function JLD2.rconvert(::Type{S}, s) where {T, S <: UniformDiagBlockSparse{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    return DiagBlockSparse(s.value, _deserialize_blockoffsets(s.ndims, s))::S
end
