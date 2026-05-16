# On-disk struct layouts for the NDTensors storage types. Used by NDTensorsJLD2Ext
# (and intended to be reusable by other serialization backends in the future). Kept in
# the main package so that the type names recorded in serialized files do not encode the
# extension module's namespace.
#
# Integer-width conventions for cross-language readability:
#   * `version::UInt32` matches `Base.VersionNumber`'s field width.
#   * Block-sparse layouts use `block_indices::Matrix{Int64}` shaped `(ndims, num_blocks)`,
#     following the COO sparse-tensor convention used by Apache Arrow, PyData Sparse, and
#     PyTorch's sparse format. The tensor rank is implicit in `size(block_indices, 1)`
#     and is preserved even when `num_blocks == 0` because HDF5 stores both matrix
#     dimensions.

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
