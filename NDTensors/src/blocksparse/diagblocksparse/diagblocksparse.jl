export DiagBlockSparse, DiagBlockSparseTensor

# DiagBlockSparse can have either Vector storage, in which case
# it is a general DiagBlockSparse tensor, or scalar storage,
# in which case the diagonal has a uniform value
# TODO: Define as an `AbstractBlockSparse`, or
# `GenericBlockSparse` parametrized by `Dense` or `Diag`.
struct DiagBlockSparse{ElT,VecT,N} <: TensorStorage{ElT}
  data::VecT
  diagblockoffsets::BlockOffsets{N}  # Block number-offset pairs

  # Nonuniform case
  function DiagBlockSparse(
    data::VecT, blockoffsets::BlockOffsets{N}
  ) where {VecT<:AbstractVector{ElT},N} where {ElT}
    return new{ElT,VecT,N}(data, blockoffsets)
  end

  # Uniform case
  function DiagBlockSparse(data::VecT, blockoffsets::BlockOffsets{N}) where {VecT<:Number,N}
    return new{VecT,VecT,N}(data, blockoffsets)
  end
end

# Data and type accessors.
blockoffsets(storage::DiagBlockSparse) = getfield(storage, :diagblockoffsets)
blockoffsetstype(storage::DiagBlockSparse) = blockoffsetstype(typeof(storage))
function blockoffsetstype(storagetype::Type{<:DiagBlockSparse})
  return fieldtype(storagetype, :diagblockoffsets)
end

# TODO: Deprecate?
diagblockoffsets(storage::DiagBlockSparse) = blockoffsets(storage)

function setdata(storagetype::Type{<:DiagBlockSparse}, data::AbstractArray)
  error("Must specify `diagblockoffsets`.")
  return DiagBlockSparse(data, blockoffsetstype(storagetype)())
end

#datatype(storage::DiagBlockSparse) = datatype(typeof(storage))
#datatype(storagetype::Type{<:DiagBlockSparse}) = fieldtype(storagetype, :data)
# TODO: Move this to a `set_types.jl` file.
# function set_datatype(
#   storagetype::Type{<:DiagBlockSparse}, datatype::Type{<:AbstractVector}
# )
#   return DiagBlockSparse{eltype(datatype),datatype,ndims(storagetype)}
# end

## DiagBlockSparse 
function DiagBlockSparse(
  ::Type{ElT}, boffs::BlockOffsets, diaglength::Integer
) where {ElT<:Number}
  return DiagBlockSparse(zeros(ElT, diaglength), boffs)
end

function DiagBlockSparse(boffs::BlockOffsets, diaglength::Integer)
  return DiagBlockSparse(Float64, boffs, diaglength)
end

function DiagBlockSparse(
  ::Type{ElT}, ::UndefInitializer, boffs::BlockOffsets, diaglength::Integer
) where {ElT<:Number}
  return DiagBlockSparse(Vector{ElT}(undef, diaglength), boffs)
end

function DiagBlockSparse(
  datatype::Type{<:AbstractArray},
  ::UndefInitializer,
  boffs::BlockOffsets,
  diaglength::Integer,
)
  return DiagBlockSparse(datatype(undef, diaglength), boffs)
end

function DiagBlockSparse(::UndefInitializer, boffs::BlockOffsets, diaglength::Integer)
  return DiagBlockSparse(Float64, undef, boffs, diaglength)
end

function findblock(
  D::DiagBlockSparse{<:Number,<:Union{Number,AbstractVector},N}, block::Block{N}; vargs...
) where {N}
  return findblock(diagblockoffsets(D), block; vargs...)
end

const NonuniformDiagBlockSparse{ElT,VecT} =
  DiagBlockSparse{ElT,VecT} where {VecT<:AbstractVector}
const UniformDiagBlockSparse{ElT,VecT} = DiagBlockSparse{ElT,VecT} where {VecT<:Number}

@propagate_inbounds function getindex(D::NonuniformDiagBlockSparse, i::Int)
  return data(D)[i]
end

getindex(D::UniformDiagBlockSparse, i::Int) = data(D)

@propagate_inbounds function setindex!(D::DiagBlockSparse, val, i::Int)
  data(D)[i] = val
  return D
end

function setindex!(D::UniformDiagBlockSparse, val, i::Int)
  return error("Cannot set elements of a uniform DiagBlockSparse storage")
end

#fill!(D::DiagBlockSparse,v) = fill!(data(D),v)

copy(D::DiagBlockSparse) = DiagBlockSparse(copy(data(D)), copy(diagblockoffsets(D)))

setdata(D::DiagBlockSparse, ndata) = DiagBlockSparse(ndata, diagblockoffsets(D))

# TODO: Move this to a `set_types.jl` file.
# TODO: Remove this once uniform diagonal tensors use FillArrays for the data.
function set_datatype(storagetype::Type{<:UniformDiagBlockSparse}, datatype::Type)
  return DiagBlockSparse{datatype,datatype,ndims(storagetype)}
end

# TODO: Make this more generic. For example, use an
# `is_composite_mutable` trait, and if `!is_composite_mutable`,
# automatically forward `NeverAlias` to `AllowAlias` since
# aliasing doesn't matter for immutable types.
function conj(::NeverAlias, storage::UniformDiagBlockSparse)
  return conj(AllowAlias(), storage)
end

## convert to complex
## TODO: this could be a generic TensorStorage function
#complex(D::DiagBlockSparse) = DiagBlockSparse(complex(data(D)), diagblockoffsets(D))

#conj(D::DiagBlockSparse{<:Real}) = D
#conj(D::DiagBlockSparse{<:Complex}) = DiagBlockSparse(conj(data(D)), diagblockoffsets(D))

# # TODO: make this generic for all storage types
# eltype(::DiagBlockSparse{ElT}) where {ElT} = ElT
# eltype(::Type{<:DiagBlockSparse{ElT}}) where {ElT} = ElT

# Deal with uniform DiagBlockSparse conversion
#convert(::Type{<:DiagBlockSparse{ElT,VecT}},D::DiagBlockSparse) where {ElT,VecT} = DiagBlockSparse(convert(VecT,data(D)))

size(D::DiagBlockSparse) = size(data(D))

# TODO: make this work for other storage besides Vector
function zeros(::Type{<:NonuniformDiagBlockSparse{ElT}}, dim::Int64) where {ElT}
  return DiagBlockSparse(zeros(ElT, dim))
end
function zeros(::Type{<:UniformDiagBlockSparse{ElT}}, dim::Int64) where {ElT}
  return DiagBlockSparse(zero(ElT))
end

#
# Type promotions involving DiagBlockSparse
# Useful for knowing how conversions should work when adding and contracting
#

function promote_rule(
  ::Type{<:UniformDiagBlockSparse{ElT1}}, ::Type{<:UniformDiagBlockSparse{ElT2}}
) where {ElT1,ElT2}
  ElR = promote_type(ElT1, ElT2)
  return DiagBlockSparse{ElR,ElR}
end

function promote_rule(
  ::Type{<:NonuniformDiagBlockSparse{ElT1,VecT1}},
  ::Type{<:NonuniformDiagBlockSparse{ElT2,VecT2}},
) where {ElT1,VecT1<:AbstractVector,ElT2,VecT2<:AbstractVector}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(VecT1, VecT2)
  return DiagBlockSparse{ElR,VecR}
end

# This is an internal definition, is there a more general way?
#promote_type(::Type{Vector{ElT1}},
#                  ::Type{ElT2}) where {ElT1<:Number,
#                                       ElT2<:Number} = Vector{promote_type(ElT1,ElT2)}
#
#promote_type(::Type{ElT1},
#                  ::Type{Vector{ElT2}}) where {ElT1<:Number,
#                                               ElT2<:Number} = promote_type(Vector{ElT2},ElT1)

# TODO: how do we make this work more generally for T2<:AbstractVector{S2}?
# Make a similartype(AbstractVector{S2},T1) -> AbstractVector{T1} function?
function promote_rule(
  ::Type{<:UniformDiagBlockSparse{ElT1,VecT1}},
  ::Type{<:NonuniformDiagBlockSparse{ElT2,Vector{ElT2}}},
) where {ElT1,VecT1<:Number,ElT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return DiagBlockSparse{ElR,VecR}
end

function promote_rule(
  ::Type{BlockSparseT1}, ::Type{<:NonuniformDiagBlockSparse{ElT2,VecT2,N2}}
) where {BlockSparseT1<:BlockSparse,ElT2<:Number,VecT2<:AbstractVector,N2}
  return promote_type(BlockSparseT1, BlockSparse{ElT2,VecT2,N2})
end

function promote_rule(
  ::Type{BlockSparseT1}, ::Type{<:UniformDiagBlockSparse{ElT2,ElT2}}
) where {BlockSparseT1<:BlockSparse,ElT2<:Number}
  return promote_type(BlockSparseT1, ElT2)
end

# Convert a DiagBlockSparse storage type to the closest Dense storage type
dense(::Type{<:NonuniformDiagBlockSparse{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}
dense(::Type{<:UniformDiagBlockSparse{ElT}}) where {ElT} = Dense{ElT,Vector{ElT}}
