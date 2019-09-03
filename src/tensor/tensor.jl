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

# This function is used in the AbstractArray interface
#Base.axes(T::Tensor) = Tuple(inds(T))

dims(ds::Dims) = ds
dim(ds::Dims) = prod(ds)

# The size is obtained from the indices
function Base.size(T::Tensor{StoreT,IndsT}) where {StoreT,IndsT}
  return dims(inds(T))
end

Base.copy(T::Tensor) = Tensor(copy(store(T)),copy(inds(T)))

function Base.similar(T::Type{Tensor{ElT,N,StoreT,IndsT}},dims) where {ElT,N,StoreT,IndsT}
  return Tensor(similar(StoreT,dims),dims)
end
#Base.similar(T::Type{<:Tensor},::Type{S}) where {S} = Tensor(similar(store(T),S),inds(T))
#Base.similar(T::Type{<:Tensor},::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)

# TODO: implement these versions
Base.similar(T::Tensor) = Tensor(similar(store(T)),inds(T))
Base.similar(T::Tensor,dims) = Tensor(similar(store(T),dims),dims)
Base.similar(T::Tensor,::Type{S}) where {S} = Tensor(similar(store(T),S),inds(T))
Base.similar(T::Tensor,::Type{S},dims) where {S} = Tensor(similar(store(T),S),dims)

function Base.convert(::Type{Tensor{ElR,N,StoreR,Inds}},
                      T::Tensor{ElT,N,StoreT,Inds}) where {ElR,ElT,N,StoreR,StoreT,Inds}
  return Tensor(convert(StoreR,store(T)),inds(T))
end

function Base.promote_rule(::Type{Tensor{ElT1,N,StoreT1,Inds}},
                           ::Type{Tensor{ElT2,N,StoreT2,Inds}}) where {ElT1,ElT2,N,StoreT1,StoreT2,Inds}
  return Tensor{promote_type(ElT1,ElT2),N,promote_type(StoreT1,StoreT2),Inds}
end

Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = Broadcast.ArrayStyle{T}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}},
                      ::Type{ElType}) where {T<:Tensor,ElType}
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

