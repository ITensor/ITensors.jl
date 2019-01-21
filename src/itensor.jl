
struct ITensor
  inds::IndexSet
  store::TensorStorage
  ITensor(is::IndexSet,st::TensorStorage) = new(is,st)
end

function ITensor(::Type{T},inds::Index...) where {T<:Number} 
  return ITensor(IndexSet(inds...),Dense{T}(dim(IndexSet(inds...))))
end

ITensor(is::IndexSet) = ITensor(Float64,is...)

ITensor(inds::Index...) = ITensor(IndexSet(inds...))

function ITensor(x::S,inds::Index...) where {S<:Number} 
  return ITensor(IndexSet(inds...),Dense{S}(x,dim(IndexSet(inds...))))
end

ITensor() = ITensor(IndexSet(),Dense{Nothing}())

inds(T::ITensor) = T.inds
store(T::ITensor) = T.store

order(T::ITensor) = order(inds(T))
dims(T::ITensor) = dims(inds(T))
dim(T::ITensor) = dim(inds(T))

copy(T::ITensor) = ITensor(copy(inds(T)),copy(store(T)))

convert(::Type{Array},T::ITensor) = storage_convert(Array,store(T),inds(T))
Array(T::ITensor) = convert(Array,T::ITensor)

getindex(T::ITensor,vals::Int...) = storage_getindex(store(T),inds(T),vals...)
function getindex(T::ITensor,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return getindex(T,vals...)
end

setindex!(T::ITensor,x::Number,vals::Int...) = storage_setindex!(store(T),inds(T),x,vals...)
function setindex!(T::ITensor,x::Number,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return setindex!(T,x,vals...)
end

function scalar(T::ITensor)
  if order(T)==0 || dim(T)==1
    return storage_scalar(store(T))
  else
    error("ITensor is not a scalar")
  end
end

randn!(T::ITensor) = storage_randn!(store(T))
function randomITensor(::Type{S},inds::Index...) where {S<:Number}
  T = ITensor(S,inds...)
  randn!(T)
  return T
end
randomITensor(inds::Index...) = randomITensor(Float64,inds...)

norm(T::ITensor) = storage_norm(store(T))
dag(T::ITensor) = ITensor(storage_dag(store(T),inds(T))...)

function permute(T::ITensor,new_inds::Index...)
  permTstore = typeof(store(T))(dim(T))
  permTinds = IndexSet(new_inds...)
  storage_permute!(permTstore,permTinds,store(T),inds(T))
  return ITensor(permTinds,permTstore)
end

function add!(A::ITensor,B::ITensor)
  storage_add!(store(A),inds(A),store(B),inds(B))
end

function +(A::ITensor,B::ITensor)
  A==B && return 2*A
  C = copy(A)
  add!(C,B)
  return C
end

function *(A::ITensor,B::ITensor)
  #TODO: Add special case of A==B
  #A==B && return ITensor(norm(A)^2)
  (Cstore,Cinds) = storage_contract(store(A),inds(A),store(B),inds(B))
  C = ITensor(Cstore,Cinds)
  return C
end

