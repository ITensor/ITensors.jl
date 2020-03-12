export Tensor,
       inds,
       ind,
       store

"""
Tensor{StoreT,IndsT}

A plain old tensor (with order independent
interface and no assumption of labels)
"""
struct Tensor{ElT,N,StoreT<:TensorStorage,IndsT} <: AbstractArray{ElT,N}
  store::StoreT
  inds::IndsT
  # The resulting Tensor is a view into the input data
  function Tensor(store::StoreT,inds::IndsT) where {StoreT<:TensorStorage{ElT},IndsT} where {ElT}
    new{ElT,ndims(IndsT),StoreT,IndsT}(store,inds)
  end
end

store(T::Tensor) = T.store
data(T::Tensor) = data(store(T))
storetype(::Tensor{ElT,N,StoreT}) where {ElT,N,StoreT} = StoreT
inds(T::Tensor) = T.inds
ind(T::Tensor,j::Integer) = inds(T)[j]
Base.eachindex(T::Tensor) = CartesianIndices(dims(inds(T)))
Base.eltype(::Tensor{ElT}) where {ElT} = ElT

#
# Generic Tensor functions
#

# The size is obtained from the indices
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))
dim(T::Tensor,i::Int) = dim(inds(T),i)
Base.size(T::Tensor) = dims(T)
Base.size(T::Tensor,i::Int) = dim(T,i)

# Needed for passing Tensor{T,2} to BLAS/LAPACK
# TODO: maybe this should only be for DenseTensor?
function Base.unsafe_convert(::Type{Ptr{ElT}},
                             T::Tensor{ElT}) where {ElT}
  return Base.unsafe_convert(Ptr{ElT},store(T))
end

Base.strides(T::Tensor) = strides(inds(T))

Base.copy(T::Tensor) = Tensor(copy(store(T)),copy(inds(T)))

Base.copyto!(R::Tensor,T::Tensor) = (copyto!(store(R),store(T)); R)

Base.complex(T::Tensor) = Tensor(complex(store(T)),copy(inds(T)))

#
# Necessary to overload since the generic fallbacks are
# slow
#

LinearAlgebra.norm(T::Tensor) = norm(store(T))

Base.conj(T::Tensor) = Tensor(conj(store(T)), copy(inds(T)))

Random.randn!(T::Tensor) = (randn!(store(T)); T)

LinearAlgebra.rmul!(T::Tensor,α::Number) = (rmul!(store(T),α); T)
scale!(T::Tensor,α::Number) = rmul!(store(T),α)

Base.fill!(T::Tensor,α::Number) = (fill!(store(T),α); T)

#function Base.similar(::Type{<:Tensor{ElT,N,StoreT}},dims) where {ElT,N,StoreT}
#  return Tensor(similar(StoreT,dim(dims)),dims)
#end

# TODO: make sure these are implemented correctly
#Base.similar(T::Type{<:Tensor},::Type{S}) where {S} = Tensor(similar(store(T),S),inds(T))
#Base.similar(T::Type{<:Tensor},::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)

Base.similar(T::Tensor) = Tensor(similar(store(T)),copy(inds(T)))

# TODO: for BlockSparse, this needs to include the offsets
# TODO: for Diag, the storage is not just the total dimension
#Base.similar(T::Tensor,dims) = _similar_from_dims(T,dims)

# To handle method ambiguity with AbstractArray
#Base.similar(T::Tensor,dims::Dims) = _similar_from_dims(T,dims)

Base.similar(T::Tensor,::Type{S}) where {S} = Tensor(similar(store(T),S),copy(inds(T)))

Base.similar(T::Tensor,::Type{S},dims) where {S<:Number} = _similar_from_dims(T,S,dims)

# To handle method ambiguity with AbstractArray
Base.similar(T::Tensor,::Type{S},dims::Dims) where {S<:Number} = _similar_from_dims(T,S,dims)

_similar_from_dims(T::Tensor,dims) = Tensor(similar(store(T),dim(dims)),dims)

function _similar_from_dims(T::Tensor,::Type{S},dims) where {S<:Number}
  return Tensor(similar(store(T),S,dim(dims)),dims)
end

function Base.convert(::Type{<:Tensor{<:Number,N,StoreR,Inds}},
                      T::Tensor{<:Number,N,<:Any,Inds}) where {N,Inds,StoreR}
  return Tensor(convert(StoreR,store(T)),copy(inds(T)))
end

function Base.zeros(::Type{<:Tensor{ElT,N,StoreT}},inds) where {ElT,N,StoreT}
  return Tensor(zeros(StoreT,dim(inds)),inds)
end

# This is to fix a method ambiguity with a Base array function
function Base.zeros(::Type{<:Tensor{ElT,N,StoreT}},inds::Dims{N}) where {ElT,N,StoreT}
  return Tensor(zeros(StoreT,dim(inds)),inds)
end

function Base.promote_rule(::Type{<:Tensor{ElT1,N1,StoreT1,IndsT1}},
                           ::Type{<:Tensor{ElT2,N2,StoreT2,IndsT2}}) where {ElT1,ElT2,
                                                                            N1,N2,
                                                                            StoreT1,StoreT2,
                                                                            IndsT1,IndsT2}
  StoreR = promote_type(StoreT1,StoreT2)
  ElR = eltype(StoreR)
  return Tensor{ElR,N3,StoreR,IndsR} where {N3,IndsR}
end

function Base.promote_rule(::Type{<:Tensor{ElT1,N,StoreT1,Inds}},
                           ::Type{<:Tensor{ElT2,N,StoreT2,Inds}}) where {ElT1,ElT2,N,
                                                                         StoreT1,StoreT2,Inds}
  StoreR = promote_type(StoreT1,StoreT2)
  ElR = eltype(StoreR)
  return Tensor{ElR,N,StoreR,Inds}
end

#function Base.promote_rule(::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}},::Type{IndsR}) where {N,ElT,StoreT,IndsR}
#  return Tensor{ElT,ndims(IndsR),StoreT,IndsR}
#end

# Convert the tensor type to the closest dense
# type
function dense(::Type{<:Tensor{ElT,NT,StoreT,IndsT}}) where {ElT,NT,StoreT,IndsT}
  return Tensor{ElT,NT,dense(StoreT),dense(IndsT)}
end

function StaticArrays.similar_type(::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}},::Type{IndsR}) where {ElT,StoreT,IndsR}
  return Tensor{ElT,ndims(IndsR),StoreT,IndsR}
end

# Convert to Array, avoiding copying if possible
array(T::Tensor) = array(dense(T))
matrix(T::Tensor{<:Number,2}) = array(T)
vector(T::Tensor{<:Number,1}) = array(T)

#
# Helper functions for BlockSparse-type storage
#

"""
nzblocks(T::Tensor)

Return a vector of the non-zero blocks of the BlockSparseTensor.
"""
nzblocks(T::Tensor) = nzblocks(store(T))

blockoffsets(T::Tensor) = blockoffsets(store(T))
nnzblocks(T::Tensor) = nnzblocks(store(T))
nnz(T::Tensor) = nnz(store(T))
nblocks(T::Tensor) = nblocks(inds(T))
blockdims(T::Tensor,block) = blockdims(inds(T),block)
blockdim(T::Tensor,block) = blockdim(inds(T),block)

"""
offset(T::Tensor,block::Block)

Get the linear offset in the data storage for the specified block.
If the specified block is not non-zero structurally, return nothing.

offset(T::Tensor,pos::Int)

Get the offset of the block at position pos
in the block-offsets list.
"""
offset(T::Tensor,block) = offset(store(T),block)

block(T::Tensor,n::Int) = block(store(T),n)

"""
blockdim(T::Tensor,pos::Int)

Get the block dimension of the block at position pos
in the block-offset list.
"""
blockdim(T::Tensor,pos::Int) = blockdim(store(T),pos)

findblock(T::Tensor,block; sorted=true) = findblock(store(T),block; sorted=sorted)

"""
isblocknz(T::Tensor,
          block::Block)

Check if the specified block is non-zero
"""
isblocknz(T::Tensor,block) = isblocknz(store(T),block)

function blockstart(T::Tensor{<:Number,N},block) where {N}
  start_index = @MVector ones(Int,N)
  for j in 1:N
    ind_j = ind(T,j)
    for block_j in 1:block[j]-1
      start_index[j] += blockdim(ind_j,block_j)
    end
  end
  return Tuple(start_index)
end

function blockend(T::Tensor{<:Number,N},
                  block) where {N}
  end_index = @MVector zeros(Int,N)
  for j in 1:N
    ind_j = ind(T,j)
    for block_j in 1:block[j]
      end_index[j] += blockdim(ind_j,block_j)
    end
  end
  return Tuple(end_index)
end

"""
blockview(T::Tensor,block::Block)

Given a specified block, return a Dense/Diag Tensor that is a view to the data
in that block.
"""
function blockview(T::Tensor,block; sorted=true)
  pos = findblock(T,block; sorted=sorted)
  return blockview(T,pos)
end

#
# Some generic getindex and setindex! functionality
#

"""
getdiagindex

Get the specified value on the diagonal
"""
function getdiagindex(T::Tensor{<:Number,N},ind::Int) where {N}
  return getindex(T,CartesianIndex(ntuple(_->ind,Val(N))))
end

"""
setdiagindex!

Set the specified value on the diagonal
"""
function setdiagindex!(T::Tensor{<:Number,N},val,ind::Int) where {N}
  setindex!(T,val,CartesianIndex(ntuple(_->ind,Val(N))))
  return T
end

#
# Some generic contraction functionality
#

function zero_contraction_output(T1::TensorT1,
                                 T2::TensorT2,
                                 indsR::IndsR) where {TensorT1<:Tensor,
                                                      TensorT2<:Tensor,
                                                      IndsR}
  return zeros(contraction_output_type(TensorT1,TensorT2,IndsR),indsR)
end

#
# Broadcasting
#

Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = Broadcast.ArrayStyle{T}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}},
                      ::Type{<:Any}) where {T<:Tensor}
  A = find_tensor(bc)
  return similar(A)
end

"`A = find_tensor(As)` returns the first Tensor among the arguments."
find_tensor(bc::Base.Broadcast.Broadcasted) = find_tensor(bc.args)
find_tensor(args::Tuple) = find_tensor(find_tensor(args[1]), Base.tail(args))
find_tensor(x) = x
find_tensor(a::Tensor, rest) = a
find_tensor(::Any, rest) = find_tensor(rest)

function Base.summary(io::IO,
                      T::Tensor)
  println(io,typeof(inds(T)))
  for (dim,ind) in enumerate(inds(T))
    println(io,"Dim $dim: ",ind)
  end
  println(io,typeof(store(T)))
  println(io," ",Base.dims2string(dims(T)))
end

#
# Printing
#

print_tensor(io::IO,T::Tensor) = Base.print_array(io,T)
print_tensor(io::IO,T::Tensor{<:Number,1}) = Base.print_array(io,reshape(T,(dim(T),1)))

