# Backend-agnostic serialization layer for NDTensors storage types.
#
# Each section below defines, for one in-memory storage type:
#   1. The [`SerializedX`](@ref) schema struct (the on-disk layout).
#   2. The [`serialized_type`](@ref) declaration mapping the in-memory type to the
#      schema struct, used by serialization backends to bridge into their own writeas /
#      type-mapping mechanism.
#   3. `Base.convert` overloads for the value-level transform in both directions.
#
# JLD2's default `wconvert` / `rconvert` already delegate to `Base.convert`, so backends
# (today only `NDTensorsJLD2Ext`) only need to register a `JLD2.writeas` declaration
# pointing at `serialized_type` — no JLD2-specific value-conversion code is required.
#
# Integer-width conventions for cross-language readability:
#   * `version::UInt32` matches `Base.VersionNumber`'s field width.
#   * Block-sparse layouts use `block_indices::Matrix{Int64}` shaped `(ndims, num_blocks)`,
#     following the COO sparse-tensor convention used by Apache Arrow, PyData Sparse, and
#     PyTorch's sparse format. The tensor rank is implicit in `size(block_indices, 1)`
#     and is preserved even when `num_blocks == 0` because HDF5 stores both matrix
#     dimensions.

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

# Shared block-offset (de)serialization helpers, used by `BlockSparse` and `DiagBlockSparse`.

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

function Base.convert(::Type{SerializedDense{T}}, d::Dense{T}) where {T}
    return SerializedDense{T}(UInt32(1), convert(Vector{T}, data(d)))
end

function Base.convert(::Type{S}, s::SerializedDense) where {T, S <: Dense{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return S(s.data)
end

# --- BlockSparse ---

"""
    SerializedBlockSparse{T}

On-disk schema for `BlockSparse{T}` storage. Version 1.

Fields:

  - `version::UInt32`
  - `data::Vector{T}` — flat element buffer.
  - `block_indices::Matrix{Int64}` — shape `(ndims, num_blocks)`, each column a block
    position (COO convention; tensor rank is `size(block_indices, 1)`).
  - `block_offsets::Vector{Int64}` — length `num_blocks`, the offset into `data` for each
    block.
"""
struct SerializedBlockSparse{T}
    version::UInt32
    data::Vector{T}
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

serialized_type(::Type{<:BlockSparse{T}}) where {T} = SerializedBlockSparse{T}

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

function Base.convert(::Type{SerializedDiag{T}}, d::NonuniformDiag{T}) where {T}
    return SerializedDiag{T}(UInt32(1), convert(Vector{T}, data(d)))
end

# Uses Diag{T} constructor because S(...) doesn't exist in NDTensors.
function Base.convert(::Type{S}, s::SerializedDiag) where {T, S <: NonuniformDiag{T}}
    eltype(s.data) === T ||
        throw(ArgumentError("data eltype mismatch: expected $T, got $(eltype(s.data))"))
    return Diag{T}(s.data)::S
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

function Base.convert(::Type{SerializedUniformDiag{T}}, d::UniformDiag{T}) where {T}
    return SerializedUniformDiag{T}(UInt32(1), convert(T, data(d)))
end

# Uses unparameterized Diag constructor because S(...) doesn't exist in NDTensors.
function Base.convert(::Type{S}, s::SerializedUniformDiag) where {T, S <: UniformDiag{T}}
    s.value isa T ||
        throw(ArgumentError("value type mismatch: expected $T, got $(typeof(s.value))"))
    return Diag(s.value)::S
end

# --- DiagBlockSparse (nonuniform) ---

"""
    SerializedDiagBlockSparse{T}

On-disk schema for non-uniform `DiagBlockSparse{T}` storage. Version 1. Layout matches
[`SerializedBlockSparse`](@ref).

Fields:

  - `version::UInt32`
  - `data::Vector{T}`
  - `block_indices::Matrix{Int64}` shape `(ndims, num_blocks)`
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

# --- DiagBlockSparse (uniform) ---

"""
    SerializedUniformDiagBlockSparse{T}

On-disk schema for uniform `DiagBlockSparse{T}` storage. Version 1.

Fields:

  - `version::UInt32`
  - `value::T` — the shared diagonal value.
  - `block_indices::Matrix{Int64}` shape `(ndims, num_blocks)`
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

function Base.convert(::Type{SerializedEmptyStorage{T}}, ::EmptyStorage{T}) where {T}
    return SerializedEmptyStorage{T}(UInt32(1))
end

Base.convert(::Type{S}, ::SerializedEmptyStorage) where {T, S <: EmptyStorage{T}} = S()
