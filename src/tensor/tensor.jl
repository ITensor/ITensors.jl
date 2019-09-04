export Tensor,
       inds,
       store

"""
Tensor{StoreT,IndsT}

A plain old tensor (with order independent
interface and no assumption of labels)
"""
struct Tensor{ElT,N,StoreT,IndsT} <: AbstractArray{ElT,N}
  store::StoreT
  inds::IndsT
  function Tensor(store::StoreT,inds::IndsT) where {StoreT,IndsT}
    new{eltype(StoreT),length(inds),StoreT,IndsT}(store,inds)
  end
end

store(T::Tensor) = T.store
inds(T::Tensor) = T.inds
ind(T::Tensor,j::Integer) = inds(T)[j]

# dim and dims are used in the Tensor interface, overload 
# base Dims here
dims(ds::Dims) = ds
dim(ds::Dims) = prod(ds)

# This is to help with some generic programming in the Tensor
# code (it helps to construct a Tuple(::NTuple{N,Int}) where the 
# only known thing for dispatch is a concrete type such
# as Dims{4})
base_type(::Type{T}) where {T<:Dims} = Dims

# The size is obtained from the indices
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))
Base.size(T::Tensor) = dims(T)

Base.copy(T::Tensor) = Tensor(copy(store(T)),copy(inds(T)))

function Base.similar(T::Type{Tensor{ElT,N,StoreT,IndsT}},dims) where {ElT,N,StoreT,IndsT}
  return Tensor(similar(StoreT,dims),dims)
end
#Base.similar(T::Type{<:Tensor},::Type{S}) where {S} = Tensor(similar(store(T),S),inds(T))
#Base.similar(T::Type{<:Tensor},::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)

# TODO: implement these versions
Base.similar(T::Tensor) = Tensor(similar(store(T)),inds(T))
Base.similar(T::Tensor,dims) = Tensor(similar(store(T),dims),dims)
# To handle method ambiguity with AbstractArray
Base.similar(T::Tensor,dims::Dims) = Tensor(similar(store(T),dims),dims)
Base.similar(T::Tensor,::Type{S}) where {S} = Tensor(similar(store(T),S),inds(T))
Base.similar(T::Tensor,::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)
# To handle method ambiguity with AbstractArray
Base.similar(T::Tensor,::Type{S},dims::Dims) where {S} = Tensor(similar(store(T),S),dims)

#function Base.convert(::Type{Tensor{<:Number,N,StoreR,Inds}},
#                      T::Tensor{<:Number,N,<:Any,Inds}) where {N,Inds,StoreR}
#  return Tensor(convert(StoreR,store(T)),inds(T))
#end

function Base.promote_rule(::Type{Tensor{ElT1,N,StoreT1,Inds}},
                           ::Type{Tensor{ElT2,N,StoreT2,Inds}}) where {ElT1,ElT2,N,StoreT1,StoreT2,Inds}
  return Tensor{promote_type(ElT1,ElT2),N,promote_type(StoreT1,StoreT2),Inds}
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

#Base.similar(T::Tensor) = Tensor(similar(store(T)),copy(inds(T)))
#Base.similar(T::Tensor,::Type{S}) where {S} = Tensor(similar(store(T)),copy(inds(T)))

#Base.getindex(A::TensorT, i::Int) where {TensorT<:Tensor} = error("getindex not yet implemented for Tensor type $TensorT")

