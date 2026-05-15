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

struct SerializedDense{T}
    version::UInt32
    data::Vector{T}
end

struct SerializedBlockSparse{T}
    version::UInt32
    data::Vector{T}
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

struct SerializedDiag{T}
    version::UInt32
    data::Vector{T}
end

struct SerializedUniformDiag{T}
    version::UInt32
    value::T
end

struct SerializedDiagBlockSparse{T}
    version::UInt32
    data::Vector{T}
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

struct SerializedUniformDiagBlockSparse{T}
    version::UInt32
    value::T
    block_indices::Matrix{Int64}
    block_offsets::Vector{Int64}
end

struct SerializedEmptyStorage{T}
    version::UInt32
end
