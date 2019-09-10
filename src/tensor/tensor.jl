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

# This may be a bad idea to overload?
# Type piracy?
Base.strides(is::Dims) = Base.size_to_strides(1, dims(is)...)

# This is to help with some generic programming in the Tensor
# code (it helps to construct a Tuple(::NTuple{N,Int}) where the 
# only known thing for dispatch is a concrete type such
# as Dims{4})
StaticArrays.similar_type(::Type{IndsT},::Val{N}) where {IndsT<:Dims,N} = Dims{N}

unioninds(is1::Dims{N1},is2::Dims{N2}) where {N1,N2} = Dims{N1+N2}(is1...,is2...)

# The size is obtained from the indices
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))
Base.size(T::Tensor) = dims(T)

Base.copy(T::Tensor) = Tensor(copy(store(T)),copy(inds(T)))

Base.complex(T::Tensor) = Tensor(complex(store(T)),copy(inds(T)))

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

function Base.promote_rule(::Type{TensorT1},
                           ::Type{TensorT2}) where 
  {TensorT1<:Tensor{ElT1,N1,StoreT1},TensorT2<:Tensor{ElT2,N2,StoreT2}} where 
  {ElT1,ElT2,N1,N2,StoreT1,StoreT2}
  return Tensor{promote_type(ElT1,ElT2),N3,promote_type(StoreT1,StoreT2)} where {N3}
end

function Base.promote_rule(::Type{Tensor{ElT1,N,StoreT1,Inds}},
                           ::Type{Tensor{ElT2,N,StoreT2,Inds}}) where {ElT1,ElT2,N,
                                                                       StoreT1,StoreT2,Inds}
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

# TODO: implement some generic fallbacks for necessary parts of the API?
#Base.getindex(A::TensorT, i::Int) where {TensorT<:Tensor} = error("getindex not yet implemented for Tensor type $TensorT")

