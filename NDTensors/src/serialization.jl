# Backend-agnostic serialization layer for NDTensors storage types.
#
# Three layers, defined once per in-memory storage type:
#
#   1. The [`SerializedX`](@ref) schema struct (the on-disk layout) plus the
#      [`serialized_type`](@ref) type-level mapping.
#   2. Permissive named functions [`serialize_convert`](@ref) and
#      [`deserialize_convert`](@ref) — the source of truth for the value-level
#      transform. `deserialize_convert` is duck-typed on its second argument so
#      it absorbs JLD2's `AbstractReconstructedType` (returned when an on-disk
#      schema struct's field layout differs from the current definition) and
#      any other shape that exposes the expected fields.
#   3. `Base.convert` overloads with typed signatures, which delegate to the
#      named functions above. These give Julia-idiomatic conversion for
#      non-JLD2 callers without polluting `Base.convert`'s namespace with the
#      permissive form.
#
# The JLD2 extension (`NDTensorsJLD2Ext`) bridges `serialized_type` into
# `JLD2.writeas` and reintroduces `JLD2.rconvert` with the permissive signature
# (delegating to `deserialize_convert`) plus a JLD2-bug idempotency workaround.
# The `JLD2.wconvert` direction uses JLD2's default delegation to `Base.convert`.
#
# Integer-width conventions for cross-language readability:
#   * `version::UInt32` matches `Base.VersionNumber`'s field width.
#   * Block-sparse layouts use `block_indices::Matrix{Int64}` shaped `(num_blocks, ndims)`,
#     i.e. each row is one block's position tuple. The tensor rank is implicit in
#     `size(block_indices, 2)` and is preserved even when `num_blocks == 0` because HDF5
#     stores both matrix dimensions.

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

"""
    NDTensors.serialize_convert(::Type{SchemaT}, x) -> SchemaT

Convert an in-memory storage value `x` to its on-disk schema representation
`SchemaT`. The source of truth for the value-level write transform; the
matching `Base.convert(::Type{SchemaT}, x)` overload is a thin shim that
delegates here.
"""
function serialize_convert end

"""
    NDTensors.deserialize_convert(::Type{InMemoryT}, s) -> InMemoryT

Convert a serialized value `s` to its in-memory form, dispatched on the target
type `InMemoryT`. The source of truth for the value-level read transform; the
matching `Base.convert(::Type{InMemoryT}, ::SerializedX)` overload is a thin
shim that delegates here.

`s` is intentionally duck-typed: in addition to the canonical `SerializedX`
schema struct, it may be a `JLD2.AbstractReconstructedType` (produced when an
on-disk schema struct's field layout differs from the current code's
definition) or any other value exposing the expected fields. Callers should
not assume `s isa SerializedX`.
"""
function deserialize_convert end

# --- Dense ---

"""
    SerializedDense{T}

On-disk schema for `Dense{T}` storage. Version 1.

Fields:

  - `version::UInt32`
  - `data::Vector{T}`
"""
struct SerializedDense{T}
    version::UInt32
    data::Vector{T}
end

serialized_type(::Type{<:Dense{T}}) where {T} = SerializedDense{T}

function serialize_convert(::Type{SerializedDense{T}}, d::Dense{T}) where {T}
    return SerializedDense{T}(UInt32(1), convert(Vector{T}, data(d)))
end

function deserialize_convert(::Type{S}, s) where {T, S <: Dense{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return S(s.data)
end

function Base.convert(::Type{SerializedDense{T}}, d::Dense{T}) where {T}
    return serialize_convert(SerializedDense{T}, d)
end
function Base.convert(::Type{S}, s::SerializedDense) where {T, S <: Dense{T}}
    return deserialize_convert(S, s)
end

# --- BlockSparse ---

"""
    SerializedBlockSparse{T}

On-disk schema for `BlockSparse{T}` storage. Version 1.

Fields:

  - `version::UInt32`
  - `data::Vector{T}` — flat element buffer.
  - `block_indices::Matrix{Int64}` — shape `(num_blocks, ndims)`, each row a block
    position. Tensor rank is `size(block_indices, 2)`.
  - `block_offsets::Vector{Int64}` — length `num_blocks`, the offset into `data` for each
    block.
"""
struct SerializedBlockSparse{T}
    version::UInt32
    data::Vector{T}
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

# Block-offset (de)serialization helpers, defined here next to `SerializedBlockSparse`
# (the first user) and reused by the `DiagBlockSparse` sections below.

function _serialize_blockoffsets(storage, ::Val{N}) where {N}
    boffs = blockoffsets(storage)
    nblocks = length(boffs)
    block_indices = Matrix{Int64}(undef, nblocks, N)
    block_offsets = Vector{Int64}(undef, nblocks)
    for (i, (block, offset)) in enumerate(pairs(boffs))
        block_indices[i, :] .= Tuple(block)
        block_offsets[i] = offset
    end
    return (; block_indices, block_offsets)
end

function _deserialize_blockoffsets(s, ::Val{N}) where {N}
    boffs = BlockOffsets{N}()
    for i in 1:size(s.block_indices, 1)
        block_tuple = NTuple{N, UInt}(@view s.block_indices[i, :])
        insert!(boffs, Block(block_tuple), s.block_offsets[i])
    end
    return boffs
end

serialized_type(::Type{<:BlockSparse{T}}) where {T} = SerializedBlockSparse{T}

function serialize_convert(
        ::Type{SerializedBlockSparse{T}}, bs::BlockSparse{T, <:Any, N}
    ) where {T, N}
    (; block_indices, block_offsets) = _serialize_blockoffsets(bs, Val(N))
    return SerializedBlockSparse{T}(
        UInt32(1), convert(Vector{T}, data(bs)), block_indices, block_offsets
    )
end

# Uses unparameterized BlockSparse constructor because S(...) doesn't exist in NDTensors.
function deserialize_convert(::Type{S}, s) where {T, S <: BlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    N = size(s.block_indices, 2)
    return BlockSparse(s.data, _deserialize_blockoffsets(s, Val(N)))::S
end

function Base.convert(
        ::Type{SerializedBlockSparse{T}}, bs::BlockSparse{T, <:Any, N}
    ) where {T, N}
    return serialize_convert(SerializedBlockSparse{T}, bs)
end
function Base.convert(::Type{S}, s::SerializedBlockSparse) where {T, S <: BlockSparse{T}}
    return deserialize_convert(S, s)
end

# --- Diag (nonuniform) ---

"""
    SerializedDiag{T}

On-disk schema for non-uniform `Diag{T}` storage. Version 1.

Fields:

  - `version::UInt32`
  - `data::Vector{T}` — the diagonal entries.
"""
struct SerializedDiag{T}
    version::UInt32
    data::Vector{T}
end

serialized_type(::Type{<:NonuniformDiag{T}}) where {T} = SerializedDiag{T}

function serialize_convert(::Type{SerializedDiag{T}}, d::NonuniformDiag{T}) where {T}
    return SerializedDiag{T}(UInt32(1), convert(Vector{T}, data(d)))
end

# Uses Diag{T} constructor because S(...) doesn't exist in NDTensors.
function deserialize_convert(::Type{S}, s) where {T, S <: NonuniformDiag{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return Diag{T}(s.data)::S
end

function Base.convert(::Type{SerializedDiag{T}}, d::NonuniformDiag{T}) where {T}
    return serialize_convert(SerializedDiag{T}, d)
end
function Base.convert(::Type{S}, s::SerializedDiag) where {T, S <: NonuniformDiag{T}}
    return deserialize_convert(S, s)
end

# --- Diag (uniform) ---

"""
    SerializedUniformDiag{T}

On-disk schema for uniform `Diag{T}` storage (all diagonal entries equal). Version 1.

Fields:

  - `version::UInt32`
  - `value::T` — the shared diagonal value.
"""
struct SerializedUniformDiag{T}
    version::UInt32
    value::T
end

serialized_type(::Type{<:UniformDiag{T}}) where {T} = SerializedUniformDiag{T}

function serialize_convert(::Type{SerializedUniformDiag{T}}, d::UniformDiag{T}) where {T}
    return SerializedUniformDiag{T}(UInt32(1), convert(T, data(d)))
end

# Uses unparameterized Diag constructor because S(...) doesn't exist in NDTensors.
function deserialize_convert(::Type{S}, s) where {T, S <: UniformDiag{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    return Diag(s.value)::S
end

function Base.convert(::Type{SerializedUniformDiag{T}}, d::UniformDiag{T}) where {T}
    return serialize_convert(SerializedUniformDiag{T}, d)
end
function Base.convert(::Type{S}, s::SerializedUniformDiag) where {T, S <: UniformDiag{T}}
    return deserialize_convert(S, s)
end

# --- DiagBlockSparse (nonuniform) ---

"""
    SerializedDiagBlockSparse{T}

On-disk schema for non-uniform `DiagBlockSparse{T}` storage. Version 1. Layout matches
[`SerializedBlockSparse`](@ref).

Fields:

  - `version::UInt32`
  - `data::Vector{T}`
  - `block_indices::Matrix{Int64}` shape `(num_blocks, ndims)`
  - `block_offsets::Vector{Int64}` length `num_blocks`
"""
struct SerializedDiagBlockSparse{T}
    version::UInt32
    data::Vector{T}
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

function serialized_type(::Type{<:NonuniformDiagBlockSparse{T}}) where {T}
    return SerializedDiagBlockSparse{T}
end

function serialize_convert(
        ::Type{SerializedDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:AbstractVector, N}
    ) where {T, N}
    (; block_indices, block_offsets) = _serialize_blockoffsets(dbs, Val(N))
    return SerializedDiagBlockSparse{T}(
        UInt32(1), convert(Vector{T}, data(dbs)), block_indices, block_offsets
    )
end

# Uses unparameterized DiagBlockSparse constructor because S(...) doesn't exist in NDTensors.
function deserialize_convert(::Type{S}, s) where {T, S <: NonuniformDiagBlockSparse{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    N = size(s.block_indices, 2)
    return DiagBlockSparse(s.data, _deserialize_blockoffsets(s, Val(N)))::S
end

function Base.convert(
        ::Type{SerializedDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:AbstractVector, N}
    ) where {T, N}
    return serialize_convert(SerializedDiagBlockSparse{T}, dbs)
end
function Base.convert(
        ::Type{S}, s::SerializedDiagBlockSparse
    ) where {T, S <: NonuniformDiagBlockSparse{T}}
    return deserialize_convert(S, s)
end

# --- DiagBlockSparse (uniform) ---

"""
    SerializedUniformDiagBlockSparse{T}

On-disk schema for uniform `DiagBlockSparse{T}` storage. Version 1.

Fields:

  - `version::UInt32`
  - `value::T` — the shared diagonal value.
  - `block_indices::Matrix{Int64}` shape `(num_blocks, ndims)`
  - `block_offsets::Vector{Int64}` length `num_blocks`
"""
struct SerializedUniformDiagBlockSparse{T}
    version::UInt32
    value::T
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

function serialized_type(::Type{<:UniformDiagBlockSparse{T}}) where {T}
    return SerializedUniformDiagBlockSparse{T}
end

function serialize_convert(
        ::Type{SerializedUniformDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:Number, N}
    ) where {T, N}
    (; block_indices, block_offsets) = _serialize_blockoffsets(dbs, Val(N))
    return SerializedUniformDiagBlockSparse{T}(
        UInt32(1), convert(T, data(dbs)), block_indices, block_offsets
    )
end

# Uses unparameterized DiagBlockSparse constructor because S(...) doesn't exist in NDTensors.
function deserialize_convert(::Type{S}, s) where {T, S <: UniformDiagBlockSparse{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    N = size(s.block_indices, 2)
    return DiagBlockSparse(s.value, _deserialize_blockoffsets(s, Val(N)))::S
end

function Base.convert(
        ::Type{SerializedUniformDiagBlockSparse{T}},
        dbs::DiagBlockSparse{T, <:Number, N}
    ) where {T, N}
    return serialize_convert(SerializedUniformDiagBlockSparse{T}, dbs)
end
function Base.convert(
        ::Type{S}, s::SerializedUniformDiagBlockSparse
    ) where {T, S <: UniformDiagBlockSparse{T}}
    return deserialize_convert(S, s)
end

# --- EmptyStorage ---

"""
    SerializedEmptyStorage{T}

On-disk schema for `EmptyStorage{T}`. Version 1. The element type `T` is carried in the
parametric type name on disk, so no data field is needed.

Fields:

  - `version::UInt32`
"""
struct SerializedEmptyStorage{T}
    version::UInt32
end

serialized_type(::Type{<:EmptyStorage{T}}) where {T} = SerializedEmptyStorage{T}

function serialize_convert(::Type{SerializedEmptyStorage{T}}, ::EmptyStorage{T}) where {T}
    return SerializedEmptyStorage{T}(UInt32(1))
end

deserialize_convert(::Type{S}, _) where {T, S <: EmptyStorage{T}} = S()

function Base.convert(::Type{SerializedEmptyStorage{T}}, e::EmptyStorage{T}) where {T}
    return serialize_convert(SerializedEmptyStorage{T}, e)
end
function Base.convert(::Type{S}, s::SerializedEmptyStorage) where {T, S <: EmptyStorage{T}}
    return deserialize_convert(S, s)
end
