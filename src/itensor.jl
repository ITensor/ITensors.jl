import Base.show

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

# This is just a stand-in for a proper delta/diag storage type
function delta(::Type{T},inds::Index...) where {T}
  d = ITensor(zero(T),inds...)
  minm = min(dims(d)...)
  for i ∈ 1:minm
    d[IndexVal.(inds,i)...] = one(T)
  end
  return d
end
delta(inds::Index...) = delta(Float64,inds...)
const δ = delta

inds(T::ITensor) = T.inds
store(T::ITensor) = T.store
eltype(T::ITensor) = eltype(store(T))

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

function commonindex(A::ITensor,B::ITensor)
  return commonindex(inds(A),inds(B))
end

# TODO: should this make a copy of the storage?
function prime(A::ITensor,vargs...)
  return ITensor(prime(inds(A),vargs...),store(A))
end

function ==(A::ITensor,B::ITensor)
  inds(A)!=inds(B) && throw(ErrorException("ITensors must have the same Indices to be equal"))
  p = calculate_permutation(inds(B),inds(A))
  for i ∈ CartesianIndices(dims(A))
    A[Tuple(i)...]≠B[Tuple(i)[p]...] && return false
  end
  return true
end

function isapprox(A::ITensor,B::ITensor;atol::Real=0.0,rtol::Real=Base.rtoldefault(eltype(A),eltype(B),atol))
  return norm(A-B) <= atol + rtol*max(norm(A),norm(B))
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

*(A::ITensor,x::Number) = A*ITensor(x)
*(x::Number,A::ITensor) = A*x

-(A::ITensor) = -one(eltype(A))*A
function +(A::ITensor,B::ITensor)
  A==B && return 2*A
  C = copy(A)
  add!(C,B)
  return C
end
-(A::ITensor,B::ITensor) = A+(-B)

function *(A::ITensor,B::ITensor)
  #TODO: Add special case of A==B
  #A==B && return ITensor(norm(A)^2)
  (Cis,Cstore) = storage_contract(store(A),inds(A),store(B),inds(B))
  C = ITensor(Cis,Cstore)
  return C
end

function show(io::IO,
              T::ITensor)
  print(io,"ITensor o=$(order(T))")
  for i = 1:order(T)
    print(io," ",inds(T)[i])
  end
  #@printf(io,"\n{%s log(scale)=%.1f}",storageTypeName(store(T)),lnum(scale(T)))
end

function svd(A::ITensor,left_inds::Index...)
  Lis = IndexSet(left_inds...)
  #TODO: make this a debug level check
  Lis⊈inds(A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  Ris = difference(inds(A),Lis)
  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,Lis...,Ris...)
  Uis,Ustore,Sis,Sstore,Vis,Vstore = storage_svd(store(A),Lis,Ris)
  return ITensor(Uis,Ustore),ITensor(Sis,Sstore),ITensor(Vis,Vstore)
end

function factorize(A::ITensor,left_inds::Index...;factorization=factorization)
  Lis = IndexSet(left_inds...)
  #TODO: make this a debug level check
  Lis⊈inds(A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  Ris = difference(inds(A),Lis)
  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,Lis...,Ris...)
  if factorization==:QR
    Qis,Qstore,Pis,Pstore = storage_qr(store(A),Lis,Ris)
  elseif factorization==:polar
    Qis,Qstore,Pis,Pstore = storage_polar(store(A),Lis,Ris)
  else
    error("Factorization $factorization not supported")
  end
  return ITensor(Qis,Qstore),ITensor(Pis,Pstore)
end

qr(A::ITensor,left_inds::Index...) = factorize(A,left_inds...;factorization=:QR)
polar(A::ITensor,left_inds::Index...) = factorize(A,left_inds...;factorization=:polar)

