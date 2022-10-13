
"""
Tensor{StoreT,IndsT}

A plain old tensor (with order independent
interface and no assumption of labels)
"""
struct Tensor{ElT,N,StoreT<:TensorStorage,IndsT} <: AbstractArray{ElT,N}
  storage::StoreT
  inds::IndsT

  """
      Tensor{ElT,N,StoreT,IndsT}(inds, store::StorageType)

  Internal constructor for creating a Tensor from the 
  storage and indices.

  The Tensor is a view of the tensor storage.

  For normal usage, use the Tensor(store::TensorStorage, inds)
  and tensor(store::TensorStorage, inds) constructors.
  """
  function Tensor{ElT,N,StoreT,IndsT}(
    ::AllowAlias, storage::TensorStorage, inds::Tuple
  ) where {ElT,N,StoreT<:TensorStorage,IndsT}
    return new{ElT,N,StoreT,IndsT}(storage, inds)
  end
end

function Tensor{ElT,N,StoreT,IndsT}(
  ::NeverAlias, storage::TensorStorage, inds
) where {ElT,N,StoreT<:TensorStorage,IndsT}
  return Tensor{ElT,N,StoreT,IndsT}(AllowAlias(), copy(storage), inds)
end

# Allow the storage and indices to be input in opposite ordering
(T::Type{<:Tensor})(as::AliasStyle, inds, storage::TensorStorage) = T(as, storage, inds)

"""
    Tensor(storage::TensorStorage, inds)

Construct a Tensor from a tensor storage and indices.
The Tensor holds a copy of the storage data.

The indices `inds` will be converted to a `Tuple`.
"""
function Tensor(as::AliasStyle, storage::TensorStorage, inds::Tuple)
  return Tensor{eltype(storage),length(inds),typeof(storage),typeof(inds)}(
    as, storage, inds
  )
end

# Automatically convert to Tuple if the indices are not a Tuple
# already (like a Vector). In the future this may be lifted
# to allow for very large tensor orders in which case Tuple
# operations may become too slow.
function Tensor(as::AliasStyle, storage::TensorStorage, inds)
  return Tensor(as, storage, Tuple(inds))
end

tensor(args...; kwargs...) = Tensor(AllowAlias(), args...; kwargs...)
Tensor(storage::TensorStorage, inds) = Tensor(NeverAlias(), storage, inds)
Tensor(inds, storage::TensorStorage) = Tensor(storage, inds)

storage(T::Tensor) = T.storage

# TODO: deprecate
store(T::Tensor) = storage(T)

data(T::Tensor) = data(storage(T))

datatype(T::Tensor) = datatype(storage(T))
datatype(tensortype::Type{<:Tensor}) = datatype(storagetype(tensortype))

indstype(::Type{<:Tensor{<:Any,<:Any,<:Any,IndsT}}) where {IndsT} = IndsT
indstype(T::Tensor) = indstype(typeof(T))

storagetype(::Type{<:Tensor{<:Any,<:Any,StoreT}}) where {StoreT} = StoreT
storagetype(T::Tensor) = storagetype(typeof(T))

# TODO: deprecate
storetype(args...) = storagetype(args...)

inds(T::Tensor) = T.inds

ind(T::Tensor, j::Integer) = inds(T)[j]

eachindex(T::Tensor) = CartesianIndices(dims(inds(T)))

eachblock(T::Tensor) = eachblock(inds(T))

eachdiagblock(T::Tensor) = eachdiagblock(inds(T))

eltype(::Tensor{ElT}) where {ElT} = ElT
scalartype(T::Tensor) = eltype(T)

strides(T::Tensor) = dim_to_strides(inds(T))

setstorage(T, nstore) = tensor(nstore, inds(T))

setinds(T, ninds) = tensor(storage(T), ninds)

#
# Generic Tensor functions
#

# The size is obtained from the indices
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))
dim(T::Tensor, i::Int) = dim(inds(T), i)
maxdim(T::Tensor) = maxdim(inds(T))
mindim(T::Tensor) = mindim(inds(T))
diaglength(T::Tensor) = mindim(T)
size(T::Tensor) = dims(T)
size(T::Tensor, i::Int) = dim(T, i)

# Needed for passing Tensor{T,2} to BLAS/LAPACK
# TODO: maybe this should only be for DenseTensor?
function unsafe_convert(::Type{Ptr{ElT}}, T::Tensor{ElT}) where {ElT}
  return unsafe_convert(Ptr{ElT}, storage(T))
end

copy(T::Tensor) = setstorage(T, copy(storage(T)))

copyto!(R::Tensor, T::Tensor) = (copyto!(storage(R), storage(T)); R)

complex(T::Tensor) = setstorage(T, complex(storage(T)))

real(T::Tensor) = setstorage(T, real(storage(T)))

imag(T::Tensor) = setstorage(T, imag(storage(T)))

# Define Base.similar in terms of NDTensors.similar
Base.similar(T::Tensor, args...) = similar(T, args...)

function map(f, x::Tensor{T}) where {T}
  if !iszero(f(zero(T)))
    error(
      "map(f, ::Tensor) currently doesn't support functions that don't preserve zeros, while you passed a function such that f(0) = $(f(zero(T))). This isn't supported right now because it doesn't necessarily preserve the sparsity structure of the input tensor.",
    )
  end
  return setstorage(x, map(f, storage(x)))
end

#
# Necessary to overload since the generic fallbacks are
# slow
#

norm(T::Tensor) = norm(storage(T))

conj(as::AliasStyle, T::Tensor) = setstorage(T, conj(as, storage(T)))
conj(T::Tensor) = conj(AllowAlias(), T)

randn!!(T::Tensor) = (randn!(T); T)
Random.randn!(T::Tensor) = (randn!(storage(T)); T)

LinearAlgebra.rmul!(T::Tensor, α::Number) = (rmul!(storage(T), α); T)
scale!(T::Tensor, α::Number) = rmul!(storage(T), α)

fill!!(T::Tensor, α::Number) = fill!(T, α)
fill!(T::Tensor, α::Number) = (fill!(storage(T), α); T)

-(T::Tensor) = setstorage(T, -storage(T))

#function similar(::Type{<:Tensor{ElT,N,StoreT}},dims) where {ElT,N,StoreT}
#  return tensor(similar(StoreT,dim(dims)),dims)
#end

# TODO: make sure these are implemented correctly
#similar(T::Type{<:Tensor},::Type{S}) where {S} = tensor(similar(storage(T),S),inds(T))
#similar(T::Type{<:Tensor},::Type{S},dims) where {S} = tensor(similar(storage(T),S),dims)

similar(T::Tensor) = setstorage(T, similar(storage(T)))

# TODO: for BlockSparse, this needs to include the offsets
# TODO: for Diag, the storage is not just the total dimension
similar(T::Tensor, dims::Tuple) = _similar_from_dims(T, dims)
function similar(::Type{TensorT}, dims::Tuple) where {TensorT<:Tensor}
  return _similar_from_dims(TensorT, dims)
end
function similar(::Type{TensorT}, dims::Tuple{}) where {TensorT<:Tensor}
  return _similar_from_dims(TensorT, dims)
end

# To handle method ambiguity with AbstractArray
#similar(T::Tensor,dims::Dims) = _similar_from_dims(T,dims)

similar(T::Tensor, ::Type{S}) where {S} = setstorage(T, similar(storage(T), S))

function similar(::Type{TensorT}, ::Type{S}, dims) where {TensorT<:Tensor,S<:Number}
  return _similar_from_dims(TensorT, S, dims)
end

similar(T::Tensor, ::Type{S}, dims) where {S<:Number} = _similar_from_dims(T, S, dims)

# To handle method ambiguity with AbstractArray
similar(T::Tensor, ::Type{S}, dims::Dims) where {S<:Number} = _similar_from_dims(T, S, dims)

_similar_from_dims(T::Tensor, dims) = tensor(similar(storage(T), dim(dims)), dims)

function _similar_from_dims(T::Tensor, ::Type{S}, dims) where {S<:Number}
  return tensor(similar(storage(T), S, dim(dims)), dims)
end

function _similar_from_dims(::Type{TensorT}, dims) where {TensorT<:Tensor}
  return _similar_from_dims(TensorT, eltype(TensorT), dims)
end

function _similar_from_dims(
  ::Type{TensorT}, ::Type{S}, dims
) where {TensorT<:Tensor,S<:Number}
  return tensor(similar(storagetype(TensorT), S, dim(dims)), dims)
end

function convert(
  ::Type{<:Tensor{<:Number,N,StoreR,Inds}}, T::Tensor{<:Number,N,<:Any,Inds}
) where {N,Inds,StoreR}
  return setstorage(T, convert(StoreR, storage(T)))
end

function zeros(TensorT::Type{<:Tensor{ElT,N,StoreT}}, inds) where {ElT,N,StoreT}
  return error("zeros(::Type{$TensorT}, inds) not implemented yet")
end

function promote_rule(
  ::Type{<:Tensor{ElT1,N1,StoreT1,IndsT1}}, ::Type{<:Tensor{ElT2,N2,StoreT2,IndsT2}}
) where {ElT1,ElT2,N1,N2,StoreT1,StoreT2,IndsT1,IndsT2}
  StoreR = promote_type(StoreT1, StoreT2)
  ElR = eltype(StoreR)
  return Tensor{ElR,N3,StoreR,IndsR} where {N3,IndsR}
end

function promote_rule(
  ::Type{<:Tensor{ElT1,N,StoreT1,Inds}}, ::Type{<:Tensor{ElT2,N,StoreT2,Inds}}
) where {ElT1,ElT2,N,StoreT1,StoreT2,Inds}
  StoreR = promote_type(StoreT1, StoreT2)
  ElR = eltype(StoreR)
  return Tensor{ElR,N,StoreR,Inds}
end

# Convert the tensor type to the closest dense
# type
function dense(::Type{<:Tensor{ElT,NT,StoreT,IndsT}}) where {ElT,NT,StoreT,IndsT}
  return Tensor{ElT,NT,dense(StoreT),IndsT}
end

dense(T::Tensor) = setstorage(T, dense(storage(T)))

function similartype(
  ::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}}, ::Type{IndsR}
) where {ElT,StoreT,IndsR}
  return Tensor{ElT,length(IndsR),StoreT,IndsR}
end

function similartype(
  ::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}}, ::Type{IndsR}
) where {ElT,StoreT,IndsR<:NTuple{NR}} where {NR}
  return Tensor{ElT,NR,StoreT,IndsR}
end

# Convert to Array, avoiding copying if possible
array(T::Tensor) = array(dense(T))
matrix(T::Tensor{<:Number,2}) = array(T)
vector(T::Tensor{<:Number,1}) = array(T)

isempty(T::Tensor) = isempty(storage(T))

#
# Helper functions for BlockSparse-type storage
#

"""
nzblocks(T::Tensor)

Return a vector of the non-zero blocks of the BlockSparseTensor.
"""
nzblocks(T::Tensor) = nzblocks(storage(T))

eachnzblock(T::Tensor) = eachnzblock(storage(T))

blockoffsets(T::Tensor) = blockoffsets(storage(T))
nnzblocks(T::Tensor) = nnzblocks(storage(T))
nnz(T::Tensor) = nnz(storage(T))
nblocks(T::Tensor) = nblocks(inds(T))
blockdims(T::Tensor, block) = blockdims(inds(T), block)
blockdim(T::Tensor, block) = blockdim(inds(T), block)

"""
offset(T::Tensor, block::Block)

Get the linear offset in the data storage for the specified block.
If the specified block is not non-zero structurally, return nothing.

offset(T::Tensor,pos::Int)

Get the offset of the block at position pos
in the block-offsets list.
"""
offset(T::Tensor, block) = offset(storage(T), block)

"""
isblocknz(T::Tensor,
          block::Block)

Check if the specified block is non-zero
"""
isblocknz(T::Tensor, block) = isblocknz(storage(T), block)

function blockstart(T::Tensor{<:Number,N}, block) where {N}
  start_index = @MVector ones(Int, N)
  for j in 1:N
    ind_j = ind(T, j)
    for block_j in 1:(block[j] - 1)
      start_index[j] += blockdim(ind_j, block_j)
    end
  end
  return Tuple(start_index)
end

function blockend(T::Tensor{<:Number,N}, block) where {N}
  end_index = @MVector zeros(Int, N)
  for j in 1:N
    ind_j = ind(T, j)
    for block_j in 1:block[j]
      end_index[j] += blockdim(ind_j, block_j)
    end
  end
  return Tuple(end_index)
end

#
# Some generic getindex and setindex! functionality
#

@propagate_inbounds @inline setindex!!(T::Tensor, x, I...) = setindex!(T, x, I...)

insertblock!!(T::Tensor, block) = insertblock!(T, block)

"""
getdiagindex

Get the specified value on the diagonal
"""
function getdiagindex(T::Tensor{<:Number,N}, ind::Int) where {N}
  return getindex(T, CartesianIndex(ntuple(_ -> ind, Val(N))))
end

# This helps make Julia's generic `diag` function work
# (at least it is necessary for `diag(::DiagTensor)`).
# TODO: Specialize this to the Tensor type, for example
# block sparse to return a block sparse vector?
function Base.similar(T::Tensor, ::Type{ElT}, dims::Tuple{Int}) where {ElT}
  return Tensor(ElT, dims)
end

"""
setdiagindex!

Set the specified value on the diagonal
"""
function setdiagindex!(T::Tensor{<:Number,N}, val, ind::Int) where {N}
  setindex!(T, val, CartesianIndex(ntuple(_ -> ind, Val(N))))
  return T
end

#
# Some generic contraction functionality
#

function zero_contraction_output(
  T1::TensorT1, T2::TensorT2, indsR::IndsR
) where {TensorT1<:Tensor,TensorT2<:Tensor,IndsR}
  return zeros(contraction_output_type(TensorT1, TensorT2, IndsR), indsR)
end

#
# Broadcasting
#

BroadcastStyle(::Type{T}) where {T<:Tensor} = Broadcast.ArrayStyle{T}()

function Base.similar(
  bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}, ::Type{ElT}
) where {T<:Tensor,ElT}
  A = find_tensor(bc)
  return similar(A, ElT)
end

"`A = find_tensor(As)` returns the first Tensor among the arguments."
find_tensor(bc::Broadcast.Broadcasted) = find_tensor(bc.args)
find_tensor(args::Tuple) = find_tensor(find_tensor(args[1]), Base.tail(args))
find_tensor(x) = x
find_tensor(a::Tensor, rest) = a
find_tensor(::Any, rest) = find_tensor(rest)

function summary(io::IO, T::Tensor)
  for (dim, ind) in enumerate(inds(T))
    println(io, "Dim $dim: ", ind)
  end
  println(io, typeof(storage(T)))
  return println(io, " ", Base.dims2string(dims(T)))
end

#
# Printing
#

print_tensor(io::IO, T::Tensor) = Base.print_array(io, T)
print_tensor(io::IO, T::Tensor{<:Number,1}) = Base.print_array(io, reshape(T, (dim(T), 1)))
