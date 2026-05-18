# Backend-agnostic serialization layer for NDTensors storage types.
#
# Pairs each storage type with its [`SerializedX`](@ref) schema struct via
# [`serialized_type`](@ref), and defines `Base.convert` overloads to do the value-level
# transform in both directions. Serialization backends (today only `NDTensorsJLD2Ext`)
# layer on top: they map their own writeas mechanism to `serialized_type`, and the
# `Base.convert` overloads here handle the actual byte-shape transform.

"""
    NDTensors.serialized_type(::Type{T}) -> Type

Return the on-disk schema struct type that an instance of `T` is serialized as.
Backend-agnostic — used by serialization extensions (e.g. `NDTensorsJLD2Ext`) to bridge
into their own writeas / type-mapping mechanism.

```julia
NDTensors.serialized_type(Dense{Float64})  # => NDTensors.SerializedDense{Float64}
```
"""
function serialized_type end

serialized_type(::Type{<:Dense{T}}) where {T} = SerializedDense{T}
serialized_type(::Type{<:BlockSparse{T}}) where {T} = SerializedBlockSparse{T}
serialized_type(::Type{<:NonuniformDiag{T}}) where {T} = SerializedDiag{T}
serialized_type(::Type{<:UniformDiag{T}}) where {T} = SerializedUniformDiag{T}
function serialized_type(::Type{<:NonuniformDiagBlockSparse{T}}) where {T}
    return SerializedDiagBlockSparse{T}
end
function serialized_type(::Type{<:UniformDiagBlockSparse{T}}) where {T}
    return SerializedUniformDiagBlockSparse{T}
end
serialized_type(::Type{<:EmptyStorage{T}}) where {T} = SerializedEmptyStorage{T}

# Shared block-offset (de)serialization helpers, used here for BlockSparse and
# DiagBlockSparse. Block positions are written as the columns of a
# `(ndims, num_blocks)` `Matrix{Int64}` (COO sparse-tensor convention).

function _serialize_blockoffsets(storage, ::Val{N}) where {N}
    boffs = blockoffsets(storage)
    nblocks = length(boffs)
    block_indices = Matrix{Int64}(undef, N, nblocks)
    block_offsets = Vector{Int64}(undef, nblocks)
    for (i, (block, offset)) in enumerate(pairs(boffs))
        block_indices[:, i] .= Tuple(block)
        block_offsets[i] = offset
    end
    return (; block_indices, block_offsets)
end

function _deserialize_blockoffsets(s, ::Val{N}) where {N}
    boffs = BlockOffsets{N}()
    for j in 1:size(s.block_indices, 2)
        block_tuple = NTuple{N, UInt}(@view s.block_indices[:, j])
        insert!(boffs, Block(block_tuple), s.block_offsets[j])
    end
    return boffs
end

# Dense

function Base.convert(::Type{SerializedDense{T}}, d::Dense{T}) where {T}
    return SerializedDense{T}(UInt32(1), convert(Vector{T}, data(d)))
end

function Base.convert(::Type{S}, s::SerializedDense) where {T, S <: Dense{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return S(s.data)
end

# BlockSparse

function Base.convert(
        ::Type{SerializedBlockSparse{T}}, bs::BlockSparse{T, <:Any, N}
    ) where {T, N}
    (; block_indices, block_offsets) = _serialize_blockoffsets(bs, Val(N))
    return SerializedBlockSparse{T}(
        UInt32(1), convert(Vector{T}, data(bs)), block_indices, block_offsets
    )
end

# Uses unparameterized BlockSparse constructor because S(...) doesn't exist in NDTensors.
function Base.convert(::Type{S}, s::SerializedBlockSparse) where {T, S <: BlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    N = size(s.block_indices, 1)
    return BlockSparse(s.data, _deserialize_blockoffsets(s, Val(N)))::S
end

# Diag (nonuniform)

function Base.convert(::Type{SerializedDiag{T}}, d::NonuniformDiag{T}) where {T}
    return SerializedDiag{T}(UInt32(1), convert(Vector{T}, data(d)))
end

# Uses Diag{T} constructor because S(...) doesn't exist in NDTensors.
function Base.convert(::Type{S}, s::SerializedDiag) where {T, S <: NonuniformDiag{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return Diag{T}(s.data)::S
end

# Diag (uniform)

function Base.convert(::Type{SerializedUniformDiag{T}}, d::UniformDiag{T}) where {T}
    return SerializedUniformDiag{T}(UInt32(1), convert(T, data(d)))
end

# Uses unparameterized Diag constructor because S(...) doesn't exist in NDTensors.
function Base.convert(::Type{S}, s::SerializedUniformDiag) where {T, S <: UniformDiag{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    return Diag(s.value)::S
end

# DiagBlockSparse (nonuniform)

function Base.convert(
        ::Type{SerializedDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:AbstractVector, N}
    ) where {T, N}
    (; block_indices, block_offsets) = _serialize_blockoffsets(dbs, Val(N))
    return SerializedDiagBlockSparse{T}(
        UInt32(1), convert(Vector{T}, data(dbs)), block_indices, block_offsets
    )
end

# Uses unparameterized DiagBlockSparse constructor because S(...) doesn't exist in NDTensors.
function Base.convert(
        ::Type{S}, s::SerializedDiagBlockSparse
    ) where {T, S <: NonuniformDiagBlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    N = size(s.block_indices, 1)
    return DiagBlockSparse(s.data, _deserialize_blockoffsets(s, Val(N)))::S
end

# DiagBlockSparse (uniform)

function Base.convert(
        ::Type{SerializedUniformDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:Number, N}
    ) where {T, N}
    (; block_indices, block_offsets) = _serialize_blockoffsets(dbs, Val(N))
    return SerializedUniformDiagBlockSparse{T}(
        UInt32(1), convert(T, data(dbs)), block_indices, block_offsets
    )
end

# Uses unparameterized DiagBlockSparse constructor because S(...) doesn't exist in NDTensors.
function Base.convert(
        ::Type{S}, s::SerializedUniformDiagBlockSparse
    ) where {T, S <: UniformDiagBlockSparse{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    N = size(s.block_indices, 1)
    return DiagBlockSparse(s.value, _deserialize_blockoffsets(s, Val(N)))::S
end

# EmptyStorage

function Base.convert(
        ::Type{SerializedEmptyStorage{T}}, ::EmptyStorage{T}
    ) where {T}
    return SerializedEmptyStorage{T}(UInt32(1))
end

function Base.convert(::Type{S}, ::SerializedEmptyStorage) where {T, S <: EmptyStorage{T}}
    return S()
end
