export Tensor,
       inds,
       store,
       data

# TODO: make into:
# abstract type TensorStorage{El} end <: AbstractVector{El}
abstract type TensorStorage end

data(S::TensorStorage) = S.data

Base.@propagate_inbounds Base.getindex(S::TensorStorage,
                                       i::Integer) = getindex(data(S),i)
Base.@propagate_inbounds Base.setindex!(S::TensorStorage,v,
                                        i::Integer) = setindex!(data(S),v,i)

Random.randn!(S::TensorStorage) = randn!(data(S))
Base.fill!(S::TensorStorage,v) = fill!(data(S),v)

Base.convert(::Type{T},D::T) where {T<:TensorStorage} = D

"""
Tensor{StoreT,IndsT}

A plain old tensor (with order independent
interface and no assumption of labels)
"""
struct Tensor{ElT,N,StoreT<:TensorStorage,IndsT} <: AbstractArray{ElT,N}
  store::StoreT
  inds::IndsT
  # The resulting Tensor is a view into the input data
  function Tensor(store::StoreT,inds::IndsT) where {StoreT<:TensorStorage,IndsT}
    new{eltype(StoreT),length(IndsT),StoreT,IndsT}(store,inds)
  end
end

store(T::Tensor) = T.store
inds(T::Tensor) = T.inds
ind(T::Tensor,j::Integer) = inds(T)[j]

#
# Tools for working with Dims/Tuples
#

# dim and dims are used in the Tensor interface, overload 
# base Dims here
dims(ds::Dims) = ds
dim(ds::Dims) = prod(ds)

Base.length(ds::Type{<:Dims{N}}) where {N} = N

# Used for BlockSparse Tensors
const BlockDims{N} = NTuple{N,NTuple{<:Any,Int}}

Base.length(ds::Type{<:BlockDims{N}}) where {N} = N

# This may be a bad idea to overload?
# Type piracy?
Base.strides(is::Dims) = Base.size_to_strides(1, dims(is)...)
Base.copy(ds::Dims) = ds

function dims(ds::BlockDims{N}) where {N}
  return ntuple(i->sum(ds[i]),Val(N))
end
function dim(ds::BlockDims{N}) where {N}
  return prod(dims(ds))
end

# A tuple of the number of blocks in each
# dimension
function nblocks(inds::BlockDims{N}) where {N}
  return ntuple(dim->length(inds[dim]),Val(N))
end

# Version taking CartestianIndex
function blockdims(inds::BlockDims{N},
                   loc) where {N}
  return ntuple(dim->inds[dim][loc[dim]],Val(N))
end

function blockindex(inds::BlockDims{N},
                      loc::Int) where {N}
  cartesian_loc = CartesianIndices(nblocks(inds))[loc]
  return Tuple(cartesian_loc)
end

# Version taking LinearIndex
function blockdims(inds::BlockDims{N},
                   loc::Int) where {N}
  # TODO: do this without conversion to CartesianIndex?
  # That may involve division and be slow?
  cartesian_loc = CartesianIndices(nblocks(inds))[loc]
  return ntuple(dim->inds[dim][cartesian_loc[dim]],Val(N))
end

function blockdim(inds::BlockDims{N},
                  loc) where {N}
  return prod(blockdims(inds,loc))
end

## TODO: should this be StaticArrays.similar_type?
#Base.promote_rule(::Type{<:Dims},
#                  ::Type{Val{N}}) where {N} = Dims{N}

ValLength(::Type{Dims{N}}) where {N} = Val{N}
ValLength(::Dims{N}) where {N} = Val{N}()

# This is to help with some generic programming in the Tensor
# code (it helps to construct a Tuple(::NTuple{N,Int}) where the 
# only known thing for dispatch is a concrete type such
# as Dims{4})
StaticArrays.similar_type(::Type{<:Dims},
                          ::Type{Val{N}}) where {N} = Dims{N}

unioninds(is1::Dims{N1},
          is2::Dims{N2}) where {N1,N2} = Dims{N1+N2}((is1...,is2...))

function deleteat(t::NTuple{N},pos::Int) where {N}
  return ntuple(i -> i < pos ? t[i] : t[i+1],Val(N-1))
end

function insertat(t::NTuple{N},
                  val::NTuple{M},
                  pos::Int) where {N,M}
  return ntuple(i -> i < pos ? t[i] :
                ( i > pos+M-1 ? t[i-1] : 
                 val[i-pos+1] ), Val(N+M-1))
end

#
# Generic Tensor functions
#

# The size is obtained from the indices
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))
Base.size(T::Tensor) = dims(T)

Base.copy(T::Tensor) = Tensor(copy(store(T)),copy(inds(T)))

Base.complex(T::Tensor) = Tensor(complex(store(T)),copy(inds(T)))

Random.randn!(T::Tensor) = (randn!(store(T)); return T)

function Base.similar(::Type{T},dims) where {T<:Tensor{<:Any,<:Any,StoreT}} where {StoreT}
  return Tensor(similar(StoreT,dims),dims)
end
# TODO: make sure these are implemented correctly
#Base.similar(T::Type{<:Tensor},::Type{S}) where {S} = Tensor(similar(store(T),S),inds(T))
#Base.similar(T::Type{<:Tensor},::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)

Base.similar(T::Tensor) = Tensor(similar(store(T)),copy(inds(T)))
Base.similar(T::Tensor,dims) = Tensor(similar(store(T),dims),dims)
# To handle method ambiguity with AbstractArray
Base.similar(T::Tensor,dims::Dims) = Tensor(similar(store(T),dims),dims)
Base.similar(T::Tensor,::Type{S}) where {S} = Tensor(similar(store(T),S),copy(inds(T)))
Base.similar(T::Tensor,::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)
# To handle method ambiguity with AbstractArray
Base.similar(T::Tensor,::Type{S},dims::Dims) where {S} = Tensor(similar(store(T),S),dims)

#function Base.convert(::Type{Tensor{<:Number,N,StoreR,Inds}},
#                      T::Tensor{<:Number,N,<:Any,Inds}) where {N,Inds,StoreR}
#  return Tensor(convert(StoreR,store(T)),copy(inds(T)))
#end

function Base.zeros(::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}},inds::IndsT) where {ElT,N,StoreT,IndsT}
  return Tensor(zeros(StoreT,dim(inds)),inds)
end

function Base.promote_rule(::Type{<:Tensor{ElT1,N1,StoreT1}},
                           ::Type{<:Tensor{ElT2,N2,StoreT2}}) where {ElT1,ElT2,N1,N2,StoreT1,StoreT2}
  return Tensor{promote_type(ElT1,ElT2),N3,promote_type(StoreT1,StoreT2)} where {N3}
end

function Base.promote_rule(::Type{Tensor{ElT1,N,StoreT1,Inds}},
                           ::Type{Tensor{ElT2,N,StoreT2,Inds}}) where {ElT1,ElT2,N,
                                                                       StoreT1,StoreT2,Inds}
  return Tensor{promote_type(ElT1,ElT2),N,promote_type(StoreT1,StoreT2),Inds}
end

#function Base.promote_rule(::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}},::Type{IndsR}) where {N,ElT,StoreT,IndsR}
#  return Tensor{ElT,length(IndsR),StoreT,IndsR}
#end

function StaticArrays.similar_type(::Type{<:Tensor{ElT,<:Any,StoreT,<:Any}},indsR) where {N,ElT,StoreT}
  return Tensor{ElT,length(indsR),StoreT,typeof(indsR)}
end

Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = Broadcast.ArrayStyle{T}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}},
                      ::Type{<:Any}) where {T<:Tensor}
  A = find_tensor(bc)
  return similar(A)
end

# This is used for overloading broadcast
"`A = find_tensor(As)` returns the first Tensor among the arguments."
find_tensor(bc::Base.Broadcast.Broadcasted) = find_tensor(bc.args)
find_tensor(args::Tuple) = find_tensor(find_tensor(args[1]), Base.tail(args))
find_tensor(x) = x
find_tensor(a::Tensor, rest) = a
find_tensor(::Any, rest) = find_tensor(rest)

# TODO: implement some generic fallbacks for necessary parts of the API?
#Base.getindex(A::TensorT, i::Int) where {TensorT<:Tensor} = error("getindex not yet implemented for Tensor type $TensorT")

