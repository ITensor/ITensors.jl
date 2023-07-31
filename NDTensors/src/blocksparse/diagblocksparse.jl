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
datatype(storage::DiagBlockSparse) = datatype(typeof(storage))
datatype(storagetype::Type{<:DiagBlockSparse}) = fieldtype(storagetype, :data)
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

# TODO: Move this to a `set_types.jl` file.
function set_datatype(
  storagetype::Type{<:DiagBlockSparse}, datatype::Type{<:AbstractVector}
)
  return DiagBlockSparse{eltype(datatype),datatype,ndims(storagetype)}
end

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

# TODO: make this generic for all storage types
eltype(::DiagBlockSparse{ElT}) where {ElT} = ElT
eltype(::Type{<:DiagBlockSparse{ElT}}) where {ElT} = ElT

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

const DiagBlockSparseTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:DiagBlockSparse}
const NonuniformDiagBlockSparseTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:NonuniformDiagBlockSparse}
const UniformDiagBlockSparseTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:UniformDiagBlockSparse}

function DiagBlockSparseTensor(
  ::Type{ElT}, ::UndefInitializer, blocks::Vector, inds
) where {ElT}
  blockoffsets, nnz = diagblockoffsets(blocks, inds)
  storage = DiagBlockSparse(ElT, undef, blockoffsets, nnz)
  return tensor(storage, inds)
end

function DiagBlockSparseTensor(::UndefInitializer, blocks::Vector, inds)
  return DiagBlockSparseTensor(Float64, undef, blocks, inds)
end

function DiagBlockSparseTensor(::Type{ElT}, blocks::Vector, inds) where {ElT}
  blockoffsets, nnz = diagblockoffsets(blocks, inds)
  storage = DiagBlockSparse(ElT, blockoffsets, nnz)
  return tensor(storage, inds)
end

DiagBlockSparseTensor(blocks::Vector, inds) = DiagBlockSparseTensor(Float64, blocks, inds)

# Uniform case
function DiagBlockSparseTensor(x::Number, blocks::Vector, inds)
  blockoffsets, nnz = diagblockoffsets(blocks, inds)
  storage = DiagBlockSparse(x, blockoffsets)
  return tensor(storage, inds)
end

diagblockoffsets(T::DiagBlockSparseTensor) = diagblockoffsets(storage(T))

"""
blockview(T::DiagBlockSparseTensor, block::Block)

Given a block in the block-offset list, return a Diag Tensor
that is a view to the data in that block (to avoid block lookup if the position
is known already).
"""
function blockview(T::DiagBlockSparseTensor, blockT::Block)
  return blockview(T, BlockOffset(blockT, offset(T, blockT)))
end

getindex(T::DiagBlockSparseTensor, block::Block) = blockview(T, block)

function blockview(T::DiagBlockSparseTensor, bof::BlockOffset)
  blockT, offsetT = bof
  blockdimsT = blockdims(T, blockT)
  blockdiaglengthT = minimum(blockdimsT)
  dataTslice = @view data(storage(T))[(offsetT + 1):(offsetT + blockdiaglengthT)]
  return tensor(Diag(dataTslice), blockdimsT)
end

function blockview(T::UniformDiagBlockSparseTensor, bof::BlockOffset)
  blockT, offsetT = bof
  blockdimsT = blockdims(T, blockT)
  return tensor(Diag(getdiagindex(T, 1)), blockdimsT)
end

IndexStyle(::Type{<:DiagBlockSparseTensor}) = IndexCartesian()

# TODO: this needs to be better (promote element type, check order compatibility,
# etc.
function convert(
  ::Type{<:DenseTensor{ElT,N}}, T::DiagBlockSparseTensor{ElT,N}
) where {ElT<:Number,N}
  return dense(T)
end

# These are rules for determining the output of a pairwise contraction of NDTensors
# (given the indices of the output tensors)
function contraction_output_type(
  TensorT1::Type{<:DiagBlockSparseTensor}, TensorT2::Type{<:BlockSparseTensor}, indsR::Tuple
)
  return similartype(promote_type(TensorT1, TensorT2), indsR)
end

function contraction_output_type(
  TensorT1::Type{<:BlockSparseTensor}, TensorT2::Type{<:DiagBlockSparseTensor}, indsR::Tuple
)
  return contraction_output_type(TensorT2, TensorT1, indsR)
end

# This performs the logic that DiagBlockSparseTensor*DiagBlockSparseTensor -> DiagBlockSparseTensor if it is not an outer
# product but -> DenseTensor if it is
# TODO: if the tensors are both order 2 (or less), or if there is an Index replacement,
# then they remain diagonal. Should we limit DiagBlockSparseTensor*DiagBlockSparseTensor to cases that
# result in a DiagBlockSparseTensor, for efficiency and type stability? What about a general
# SparseTensor result?
function contraction_output_type(
  TensorT1::Type{<:DiagBlockSparseTensor{<:Number,N1}},
  TensorT2::Type{<:DiagBlockSparseTensor{<:Number,N2}},
  indsR::Tuple,
) where {N1,N2}
  if ValLength(indsR) === Val{N1 + N2}
    # Turn into is_outer(inds1,inds2,indsR) function?
    # How does type inference work with arithmatic of compile time values?
    return similartype(dense(promote_type(TensorT1, TensorT2)), indsR)
  end
  return similartype(promote_type(TensorT1, TensorT2), indsR)
end

# The output must be initialized as zero since it is sparse, cannot be undefined
function contraction_output(T1::DiagBlockSparseTensor, T2::Tensor, indsR)
  return zero_contraction_output(T1, T2, indsR)
end
function contraction_output(T1::Tensor, T2::DiagBlockSparseTensor, indsR)
  return contraction_output(T2, T1, indsR)
end

# function contraction_output(T1::DiagBlockSparseTensor, T2::DiagBlockSparseTensor, indsR)
#   return zero_contraction_output(T1, T2, indsR)
# end

# Determine the contraction output and block contractions
function contraction_output(
  tensor1::DiagBlockSparseTensor,
  labelstensor1,
  tensor2::DiagBlockSparseTensor,
  labelstensor2,
  labelsR,
)
  indsR = contract_inds(inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsR)
  TensorR = contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
  blockoffsetsR, contraction_plan = contract_blockoffsets(
    blockoffsets(tensor1),
    inds(tensor1),
    labelstensor1,
    blockoffsets(tensor2),
    inds(tensor2),
    labelstensor2,
    indsR,
    labelsR,
  )
  R = similar(TensorR, blockoffsetsR, indsR)
  return R # , contraction_plan
end

## TODO: Is there a way to make this generic?
# NDTensors.similar
function similar(
  tensortype::Type{<:DiagBlockSparseTensor}, blockoffsets::BlockOffsets, dims::Tuple
)
  return Tensor(similar(storagetype(tensortype), blockoffsets, dims), dims)
end

# NDTensors.similar
function similar(
  storagetype::Type{<:DiagBlockSparse}, blockoffsets::BlockOffsets, dims::Tuple
)
  # TODO: Improve this with FillArrays.jl
  # data = similar(datatype(storagetype), nnz(blockoffsets, dims))
  data = zero(datatype(storagetype))
  return DiagBlockSparse(data, blockoffsets)
end

function array(T::DiagBlockSparseTensor{ElT,N}) where {ElT,N}
  return array(dense(T))
end
matrix(T::DiagBlockSparseTensor{<:Number,2}) = array(T)
vector(T::DiagBlockSparseTensor{<:Number,1}) = array(T)

function Array{ElT,N}(T::DiagBlockSparseTensor{ElT,N}) where {ElT,N}
  return array(T)
end

function Array(T::DiagBlockSparseTensor{ElT,N}) where {ElT,N}
  return Array{ElT,N}(T)
end

getdiagindex(T::DiagBlockSparseTensor{<:Number}, ind::Int) = storage(T)[ind]

# XXX: handle case of missing diagonal blocks
function setdiagindex!(T::DiagBlockSparseTensor{<:Number}, val, ind::Int)
  storage(T)[ind] = val
  return T
end

function setdiag(T::DiagBlockSparseTensor, val, ind::Int)
  return tensor(DiagBlockSparse(val), inds(T))
end

function setdiag(T::UniformDiagBlockSparseTensor, val, ind::Int)
  return tensor(DiagBlockSparse(val, blockoffsets(T)), inds(T))
end

@propagate_inbounds function getindex(
  T::DiagBlockSparseTensor{ElT,N}, inds::Vararg{Int,N}
) where {ElT,N}
  if all(==(inds[1]), inds)
    return storage(T)[inds[1]]
  else
    return zero(eltype(ElT))
  end
end

@propagate_inbounds function getindex(T::DiagBlockSparseTensor{<:Number,1}, ind::Int)
  return storage(T)[ind]
end

@propagate_inbounds function getindex(T::DiagBlockSparseTensor{<:Number,0})
  return storage(T)[1]
end

# Set diagonal elements
# Throw error for off-diagonal
@propagate_inbounds function setindex!(
  T::DiagBlockSparseTensor{<:Number,N}, val, inds::Vararg{Int,N}
) where {N}
  all(==(inds[1]), inds) ||
    error("Cannot set off-diagonal element of DiagBlockSparse storage")
  storage(T)[inds[1]] = val
  return T
end

@propagate_inbounds function setindex!(T::DiagBlockSparseTensor{<:Number,1}, val, ind::Int)
  storage(T)[ind] = val
  return T
end

@propagate_inbounds function setindex!(T::DiagBlockSparseTensor{<:Number,0}, val)
  storage(T)[1] = val
  return T
end

function setindex!(
  T::UniformDiagBlockSparseTensor{<:Number,N}, val, inds::Vararg{Int,N}
) where {N}
  return error("Cannot set elements of a uniform DiagBlockSparse storage")
end

# TODO: make a fill!! that works for uniform and non-uniform
#fill!(T::DiagBlockSparseTensor,v) = fill!(storage(T),v)

function dense(
  ::Type{<:Tensor{ElT,N,StoreT,IndsT}}
) where {ElT,N,StoreT<:DiagBlockSparse,IndsT}
  return Tensor{ElT,N,dense(StoreT),IndsT}
end

# convert to Dense
function dense(T::TensorT) where {TensorT<:DiagBlockSparseTensor}
  R = zeros(dense(TensorT), inds(T))
  for i in 1:diaglength(T)
    setdiagindex!(R, getdiagindex(T, i), i)
  end
  return R
end

# convert to BlockSparse
function denseblocks(D::Tensor)
  nzblocksD = nzblocks(D)
  T = BlockSparseTensor(eltype(D), nzblocksD, inds(D))
  for b in nzblocksD
    for n in 1:diaglength(D)
      setdiagindex!(T, getdiagindex(D, n), n)
    end
  end
  return T
end

function outer!(
  R::DenseTensor{<:Number,NR},
  T1::DiagBlockSparseTensor{<:Number,N1},
  T2::DiagBlockSparseTensor{<:Number,N2},
) where {NR,N1,N2}
  for i1 in 1:diaglength(T1), i2 in 1:diaglength(T2)
    indsR = CartesianIndex{NR}(ntuple(r -> r ≤ N1 ? i1 : i2, Val(NR)))
    R[indsR] = getdiagindex(T1, i1) * getdiagindex(T2, i2)
  end
  return R
end

# TODO: write an optimized version of this?
function outer!(R::DenseTensor{ElR}, T1::DenseTensor, T2::DiagBlockSparseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R, T1, dense(T2))
  return R
end

function outer!(R::DenseTensor{ElR}, T1::DiagBlockSparseTensor, T2::DenseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R, dense(T1), T2)
  return R
end

# Right an in-place version
function outer(
  T1::DiagBlockSparseTensor{ElT1,N1}, T2::DiagBlockSparseTensor{ElT2,N2}
) where {ElT1,ElT2,N1,N2}
  indsR = unioninds(inds(T1), inds(T2))
  R = tensor(Dense(zeros(promote_type(ElT1, ElT2), dim(indsR))), indsR)
  outer!(R, T1, T2)
  return R
end

function permutedims!(
  R::DiagBlockSparseTensor{<:Number,N},
  T::DiagBlockSparseTensor{<:Number,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  # TODO: check that inds(R)==permute(inds(T),perm)?
  for i in 1:diaglength(R)
    @inbounds setdiagindex!(R, f(getdiagindex(R, i), getdiagindex(T, i)), i)
  end
  return R
end

function permutedims(
  T::UniformDiagBlockSparseTensor{ElT,N}, perm::NTuple{N,Int}, f::Function=identity
) where {ElT,N}
  R = tensor(DiagBlockSparse(f(getdiagindex(T, 1))), permute(inds(T), perm))
  return R
end

# Version that may overwrite in-place or may return the result
function permutedims!!(
  R::NonuniformDiagBlockSparseTensor{<:Number,N},
  T::NonuniformDiagBlockSparseTensor{<:Number,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  permutedims!(RR, T, perm, f)
  return RR
end

function permutedims!!(
  R::UniformDiagBlockSparseTensor{ElR,N},
  T::UniformDiagBlockSparseTensor{ElT,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {ElR,ElT,N}
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  RR = tensor(DiagBlockSparse(f(getdiagindex(RR, 1), getdiagindex(T, 1))), inds(RR))
  return RR
end

function permutedims!(
  R::DenseTensor{ElR,N},
  T::DiagBlockSparseTensor{ElT,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {ElR,ElT,N}
  for i in 1:diaglength(T)
    @inbounds setdiagindex!(R, f(getdiagindex(R, i), getdiagindex(T, i)), i)
  end
  return R
end

function permutedims!!(
  R::DenseTensor{ElR,N},
  T::DiagBlockSparseTensor{ElT,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {ElR,ElT,N}
  permutedims!(R, T, perm, f)
  return R
end

function _contract!!(
  R::UniformDiagBlockSparseTensor{ElR,NR},
  labelsR,
  T1::UniformDiagBlockSparseTensor{<:Number,N1},
  labelsT1,
  T2::UniformDiagBlockSparseTensor{<:Number,N2},
  labelsT2,
) where {ElR,NR,N1,N2}
  if NR == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    R = setdiag(R, diaglength(T1) * getdiagindex(T1, 1) * getdiagindex(T2, 1), 1)
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    R = setdiag(R, getdiagindex(T1, 1) * getdiagindex(T2, 1), 1)
  end
  return R
end

# TODO: Improve this with FillArrays.jl
norm(S::UniformDiagBlockSparseTensor) = sqrt(mindim(S) * abs2(data(S)))

function contraction_output(
  T1::TensorT1, labelsT1, T2::TensorT2, labelsT2, labelsR
) where {TensorT1<:BlockSparseTensor,TensorT2<:DiagBlockSparseTensor}
  indsR = contract_inds(inds(T1), labelsT1, inds(T2), labelsT2, labelsR)
  TensorR = contraction_output_type(TensorT1, TensorT2, indsR)
  blockoffsetsR, contraction_plan = contract_blockoffsets(
    blockoffsets(T1),
    inds(T1),
    labelsT1,
    blockoffsets(T2),
    inds(T2),
    labelsT2,
    indsR,
    labelsR,
  )
  R = zeros(TensorR, blockoffsetsR, indsR)
  return R, contraction_plan
end

function contract(
  T1::BlockSparseTensor,
  labelsT1,
  T2::DiagBlockSparseTensor,
  labelsT2,
  labelsR=contract_labels(labelsT1, labelsT2),
)
  R, contraction_plan = contraction_output(T1, labelsT1, T2, labelsT2, labelsR)
  R = contract!(R, labelsR, T1, labelsT1, T2, labelsT2, contraction_plan)
  return R
end

function contract(
  T1::DiagBlockSparseTensor,
  labelsT1,
  T2::BlockSparseTensor,
  labelsT2,
  labelsR=contract_labels(labelsT2, labelsT1),
)
  return contract(T2, labelsT2, T1, labelsT1, labelsR)
end

function contract!(
  R::BlockSparseTensor{ElR,NR},
  labelsR,
  T1::BlockSparseTensor,
  labelsT1,
  T2::DiagBlockSparseTensor,
  labelsT2,
  contraction_plan,
) where {ElR<:Number,NR}
  already_written_to = Dict{Block{NR},Bool}()
  indsR = inds(R)
  indsT1 = inds(T1)
  indsT2 = inds(T2)
  # In R .= α .* (T1 * T2) .+ β .* R
  α = one(ElR)
  for (block1, block2, blockR) in contraction_plan
    T1block = T1[block1]
    T2block = T2[block2]
    Rblock = R[blockR]

    # <fermions>
    α = compute_alpha(
      ElR, labelsR, blockR, indsR, labelsT1, block1, indsT1, labelsT2, block2, indsT2
    )

    β = one(ElR)
    if !haskey(already_written_to, blockR)
      already_written_to[blockR] = true
      # Overwrite the block of R
      β = zero(ElR)
    end
    contract!(Rblock, labelsR, T1block, labelsT1, T2block, labelsT2, α, β)
  end
  return R
end

function contract!(
  C::BlockSparseTensor,
  Clabels,
  A::BlockSparseTensor,
  Alabels,
  B::DiagBlockSparseTensor,
  Blabels,
)
  return contract!(C, Clabels, B, Blabels, A, Alabels)
end

function show(io::IO, mime::MIME"text/plain", T::DiagBlockSparseTensor)
  summary(io, T)
  for (n, block) in enumerate(keys(diagblockoffsets(T)))
    blockdimsT = blockdims(T, block)
    println(io, block)
    println(io, " [", _range2string(blockstart(T, block), blockend(T, block)), "]")
    print_tensor(io, blockview(T, block))
    n < nnzblocks(T) && print(io, "\n\n")
  end
end

show(io::IO, T::DiagBlockSparseTensor) = show(io, MIME("text/plain"), T)
