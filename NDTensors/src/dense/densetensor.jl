#
# DenseTensor (Tensor using Dense storage)
#

const DenseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Dense}

DenseTensor(::Type{ElT}, inds) where {ElT} = tensor(Dense(ElT, dim(inds)), inds)

# Special convenience function for Int
# dimensions
DenseTensor(::Type{ElT}, inds::Int...) where {ElT} = DenseTensor(ElT, inds)

DenseTensor(inds) = tensor(Dense(dim(inds)), inds)

DenseTensor(inds::Int...) = DenseTensor(inds)

function DenseTensor(::Type{ElT}, ::UndefInitializer, inds) where {ElT}
  return tensor(Dense(ElT, undef, dim(inds)), inds)
end

function DenseTensor(::Type{ElT}, ::UndefInitializer, inds::Int...) where {ElT}
  return DenseTensor(ElT, undef, inds)
end

DenseTensor(::UndefInitializer, inds) = tensor(Dense(undef, dim(inds)), inds)

DenseTensor(::UndefInitializer, inds::Int...) = DenseTensor(undef, inds)

#
# Random constructors
#

function randomDenseTensor(::Type{ElT}, inds) where {ElT}
  return tensor(generic_randn(Dense{ElT}, dim(inds)), inds)
end

randomDenseTensor(inds) = randomDenseTensor(default_eltype(), inds)

## End Random Dense Tensor constructor

# Basic functionality for AbstractArray interface
IndexStyle(::Type{<:DenseTensor}) = IndexLinear()

# Override CartesianIndices iteration to iterate
# linearly through the Dense storage (faster)
iterate(T::DenseTensor, args...) = iterate(storage(T), args...)

function _zeros(TensorT::Type{<:DenseTensor}, inds)
  return tensor(generic_zeros(storagetype(TensorT), dim(inds)), inds)
end

function zeros(TensorT::Type{<:DenseTensor}, inds)
  return _zeros(TensorT, inds)
end

# To fix method ambiguity with zeros(::Type, ::Tuple)
function zeros(TensorT::Type{<:DenseTensor}, inds::Dims)
  return _zeros(TensorT, inds)
end

function zeros(TensorT::Type{<:DenseTensor}, inds::Tuple{})
  return _zeros(TensorT, inds)
end

convert(::Type{Array}, T::DenseTensor) = reshape(data(storage(T)), dims(inds(T)))

# Create an Array that is a view of the Dense Tensor
# Useful for using Base Array functions
array(T::DenseTensor) = convert(Array, T)

function Array{ElT,N}(T::DenseTensor{ElT,N}) where {ElT,N}
  return copy(array(T))
end

function Array(T::DenseTensor{ElT,N}) where {ElT,N}
  return Array{ElT,N}(T)
end

#
# Single index
#

@propagate_inbounds function getindex(T::DenseTensor{<:Number}, I::Integer...)
  Base.@_inline_meta
  return getindex(data(T), Base._sub2ind(T, I...))
end

@propagate_inbounds function getindex(T::DenseTensor{<:Number}, I::CartesianIndex)
  Base.@_inline_meta
  return getindex(T, I.I...)
end

@propagate_inbounds function setindex!(
  T::DenseTensor{<:Number}, x::Number, I::Vararg{Integer}
)
  Base.@_inline_meta
  setindex!(data(T), x, Base._sub2ind(T, I...))
  return T
end

@propagate_inbounds function setindex!(
  T::DenseTensor{<:Number}, x::Number, I::CartesianIndex
)
  Base.@_inline_meta
  setindex!(T, x, I.I...)
  return T
end

#
# Linear indexing
#

@propagate_inbounds @inline getindex(T::DenseTensor, i::Integer) = storage(T)[i]

@propagate_inbounds @inline function setindex!(T::DenseTensor, v, i::Integer)
  return (storage(T)[i] = v; T)
end

#
# Slicing
# TODO: this doesn't allow colon right now
# Create a DenseView that stores a Dense and an offset?
#

## @propagate_inbounds function _getindex(
##   T::DenseTensor{ElT,N}, I::CartesianIndices{N}
## ) where {ElT,N}
##   storeR = Dense(vec(@view array(T)[I]))
##   indsR = Tuple(I[end] - I[1] + CartesianIndex(ntuple(_ -> 1, Val(N))))
##   return tensor(storeR, indsR)
## end
## 
## @propagate_inbounds function getindex(T::DenseTensor{ElT,N}, I...) where {ElT,N}
##   return _getindex(T, CartesianIndices(I))
## end

@propagate_inbounds function getindex(T::DenseTensor, I...)
  AI = @view array(T)[I...]
  storeR = Dense(vec(AI))
  indsR = size(AI)
  return tensor(storeR, indsR)
end

# Reshape a DenseTensor using the specified dimensions
# This returns a view into the same Tensor data
function reshape(T::DenseTensor, dims)
  dim(T) == dim(dims) || error("Total new dimension must be the same as the old dimension")
  return tensor(storage(T), dims)
end

# This version fixes method ambiguity with AbstractArray reshape
function reshape(T::DenseTensor, dims::Dims)
  dim(T) == dim(dims) || error("Total new dimension must be the same as the old dimension")
  return tensor(storage(T), dims)
end

function reshape(T::DenseTensor, dims::Int...)
  return tensor(storage(T), tuple(dims...))
end

# If the storage data are regular Vectors, use Base.copyto!
function copyto!(
  R::Tensor{<:Number,N,<:Dense{<:Number,<:Vector}},
  T::Tensor{<:Number,N,<:Dense{<:Number,<:Vector}},
) where {N}
  RA = array(R)
  TA = array(T)
  RA .= TA
  return R
end

# If they are something more complicated like views, use Strided copyto!
function copyto!(
  R::DenseTensor{<:Number,N,StoreT}, T::DenseTensor{<:Number,N,StoreT}
) where {N,StoreT<:StridedArray}
  RA = array(R)
  TA = array(T)
  @strided RA .= TA
  return R
end

# Maybe allocate output data.
# TODO: Remove this in favor of `map!`
# applied to `PermutedDimsArray`.
function permutedims!!(R::DenseTensor, T::DenseTensor, perm, f::Function=(r, t) -> t)
  Base.checkdims_perm(R, T, perm)
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  permutedims!(RR, T, perm, f)
  return RR
end

# TODO: call permutedims!(R,T,perm,(r,t)->t)?
function permutedims!(
  R::DenseTensor{<:Number,N,StoreT}, T::DenseTensor{<:Number,N,StoreT}, perm::NTuple{N,Int}
) where {N,StoreT<:StridedArray}
  RA = array(R)
  TA = array(T)
  @strided RA .= permutedims(TA, perm)
  return R
end

function copyto!(R::DenseTensor{<:Number,N}, T::DenseTensor{<:Number,N}) where {N}
  RA = array(R)
  TA = array(T)
  RA .= TA
  return R
end

# TODO: call permutedims!(R,T,perm,(r,t)->t)?
function permutedims!(
  R::DenseTensor{<:Number,N}, T::DenseTensor{<:Number,N}, perm::NTuple{N,Int}
) where {N}
  RA = array(R)
  TA = array(T)
  RA .= permutedims(TA, perm)
  return R
end

function apply!(
  R::DenseTensor{<:Number,N,StoreT},
  T::DenseTensor{<:Number,N,StoreT},
  f::Function=(r, t) -> t,
) where {N,StoreT<:StridedArray}
  RA = array(R)
  TA = array(T)
  @strided RA .= f.(RA, TA)
  return R
end

function apply!(R::DenseTensor, T::DenseTensor, f::Function=(r, t) -> t)
  RA = array(R)
  TA = array(T)
  RA .= f.(RA, TA)
  return R
end

function permutedims!(
  R::DenseTensor{<:Number,N}, T::DenseTensor{<:Number,N}, perm, f::Function
) where {N}
  if nnz(R) == 1 && nnz(T) == 1
    R[1] = f(R[1], T[1])
    return R
  end
  RA = array(R)
  TA = array(T)
  @strided RA .= f.(RA, permutedims(TA, perm))
  return R
end

"""
    NDTensors.permute_reshape(T::Tensor,pos...)

Takes a permutation that is split up into tuples. Index positions
within the tuples are combined.

For example:

permute_reshape(T,(3,2),1)

First T is permuted as `permutedims(3,2,1)`, then reshaped such
that the original indices 3 and 2 are combined.
"""
function permute_reshape(
  T::DenseTensor{ElT,NT,IndsT}, pos::Vararg{Any,N}
) where {ElT,NT,IndsT,N}
  perm = flatten(pos...)

  length(perm) ≠ NT && error("Index positions must add up to order of Tensor ($N)")
  isperm(perm) || error("Index positions must be a permutation")

  dimsT = dims(T)
  indsT = inds(T)
  if !is_trivial_permutation(perm)
    T = permutedims(T, perm)
  end
  if all(p -> length(p) == 1, pos) && N == NT
    return T
  end
  newdims = MVector(ntuple(_ -> eltype(IndsT)(1), Val(N)))
  for i in 1:N
    if length(pos[i]) == 1
      # No reshape needed, just use the
      # original index
      newdims[i] = indsT[pos[i][1]]
    else
      newdim_i = 1
      for p in pos[i]
        newdim_i *= dimsT[p]
      end
      newdims[i] = eltype(IndsT)(newdim_i)
    end
  end
  newinds = similartype(IndsT, Val{N})(Tuple(newdims))
  return reshape(T, newinds)
end

function show(io::IO, mime::MIME"text/plain", T::DenseTensor)
  summary(io, T)
  return print_tensor(io, T)
end
