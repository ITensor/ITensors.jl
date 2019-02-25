
struct ITensor
  inds::IndexSet
  store::TensorStorage
  #TODO: check that the storage is consistent with the
  #total dimension of the indices
  ITensor(is::IndexSet,st::TensorStorage) = new(is,st)
end

function ITensor(::Type{T},inds::IndexSet) where {T<:Number}
  return ITensor(inds,Dense{T}(dim(inds)))
end
ITensor(::Type{T},inds::Index...) where {T<:Number} = ITensor(T,IndexSet(inds...))

ITensor(is::IndexSet) = ITensor(Float64,is...)
ITensor(inds::Index...) = ITensor(IndexSet(inds...))

function ITensor(x::S,inds::IndexSet) where {S<:Number}
  return ITensor(inds,Dense{S}(x,dim(inds)))
end
ITensor(x::S,inds::Index...) where {S<:Number} = ITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function ITensor(A::Array{S},inds::IndexSet) where {S<:Number}
  return ITensor(inds,Dense{S}(A))
end
ITensor(A::Array{S},inds::Index...) where {S<:Number} = ITensor(A,IndexSet(inds...))

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

#convert(::Type{Array},T::ITensor) = storage_convert(Array,store(T),inds(T))
Array(T::ITensor) = storage_convert(Array,store(T),inds(T))

function getindex(T::ITensor,vals::Int...) 
  order(T) ≠ length(vals) && error("In getindex(::ITensor,::Int..), number of values provided ($(length(vals))) must equal order of ITensor ($(order(T)))")
  storage_getindex(store(T),inds(T),vals...)
end
function getindex(T::ITensor,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return getindex(T,vals...)
end
getindex(T::ITensor, ivs::Vector{IndexVal}) = [T[iv] for iv in ivs] 

setindex!(T::ITensor,x::Number,vals::Int...) = storage_setindex!(store(T),inds(T),x,vals...)
function setindex!(T::ITensor,x::Number,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return setindex!(T,x,vals...)
end
function setindex!(T::ITensor,x::Vector{<:Number}, ivs::Union{IndexVal, AbstractVector{IndexVal}}...)
    p = calculate_permutation(inds(T),map(x->x isa IndexVal ? x : x[1], ivs))
    vals = map(x->x isa IndexVal ? val(x) : val.(x), ivs)
    storage_setindex!(store(T),inds(T),x,vals...)
end

findindex(A::ITensor,ts::String) = findindex(inds(A),ts)

function commonindex(A::ITensor,B::ITensor)
  return commonindex(inds(A),inds(B))
end
function commoninds(A::ITensor,B::ITensor)
  return inds(A)∩inds(B)
end

hasindex(T::ITensor,I::Index) = hasindex(inds(T),I)

# TODO: should this make a copy of the storage?
function prime(A::ITensor,vargs...)
  return ITensor(prime(inds(A),vargs...),store(A))
end
adjoint(A::ITensor) = prime(A)
function primeexcept(A::ITensor,vargs...)
  return ITensor(primeexcept(inds(A),vargs...),store(A))
end
function setprime(A::ITensor,vargs...)
  return ITensor(setprime(inds(A),vargs...),store(A))
end
function noprime(A::ITensor,vargs...)
  return ITensor(noprime(inds(A),vargs...),store(A))
end
function mapprime(A::ITensor,vargs...)
  return ITensor(mapprime(inds(A),vargs...),store(A))
end
function swapprime(A::ITensor,vargs...)
  return ITensor(swapprime(inds(A),vargs...),store(A))
end

function addtags(A::ITensor,vargs...)
  return ITensor(addtags(inds(A),vargs...),store(A))
end

function removetags(A::ITensor,vargs...)
  return ITensor(removetags(inds(A),vargs...),store(A))
end

function replacetags(A::ITensor,vargs...)
  return ITensor(replacetags(inds(A),vargs...),store(A))
end

function swaptags(A::ITensor,vargs...)
  return ITensor(swaptags(inds(A),vargs...),store(A))
end

function tags(A::ITensor,vargs...)
  return ITensor(tags(inds(A),vargs...),store(A))
end

function ==(A::ITensor,B::ITensor)::Bool
  inds(A)!=inds(B) && throw(ErrorException("ITensors must have the same Indices to be equal"))
  p = calculate_permutation(inds(B),inds(A))
  for i ∈ CartesianIndices(dims(A))
    A[Tuple(i)...]≠B[Tuple(i)[p]...] && return false
  end
  return true
end

function isapprox(A::ITensor,
                  B::ITensor;
                  atol::Real=0.0,
                  rtol::Real=Base.rtoldefault(eltype(A),eltype(B),atol))
  return norm(A-B) <= atol + rtol*max(norm(A),norm(B))
end

function scalar(T::ITensor)
  if !(order(T)==0 || dim(T)==1)
    error("ITensor is not a scalar")
  end
  return storage_scalar(store(T))
end

randn!(T::ITensor) = storage_randn!(store(T))

function randomITensor(::Type{S},inds::IndexSet) where {S<:Number}
  T = ITensor(S,inds)
  randn!(T)
  return T
end
randomITensor(::Type{S},inds::Index...) where {S<:Number} = randomITensor(S,IndexSet(inds...))
randomITensor(inds::IndexSet) = randomITensor(Float64,inds)
randomITensor(inds::Index...) = randomITensor(Float64,IndexSet(inds...))

norm(T::ITensor) = storage_norm(store(T))
dag(T::ITensor) = ITensor(storage_dag(store(T),inds(T))...)

function permute(T::ITensor,permTinds::IndexSet)
  permTstore = typeof(store(T))(dim(T))
  storage_permute!(permTstore,permTinds,store(T),inds(T))
  return ITensor(permTinds,permTstore)
end
permute(T::ITensor,new_inds::Index...) = permute(T,IndexSet(new_inds...))

function add!(A::ITensor,B::ITensor)
  storage_add!(store(A),inds(A),store(B),inds(B))
end

#TODO: improve these using a storage_mult call
*(A::ITensor,x::Number) = A*ITensor(x)
*(x::Number,A::ITensor) = A*x

/(A::ITensor,x::Number) = A*ITensor(1.0/x)

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

function findtags(T::ITensor,
                  tags::String)
  ts = TagSet(tags)
  for i in inds(T)
    if hastags(i,ts)
      return i
    end
  end
  error("findtags: ITensor has no Index with given tags: $ts")
  return Index()
end

function findinds(T::ITensor,
                  tags::String)
  vinds = Index[]
  ts = TagSet(tags)
  for i in inds(T)
    if hastags(i,ts)
      push!(vinds,i)
    end
  end
  return vinds
end

function eigen(A::ITensor,
               left_inds::NTuple{NL,Index},
               right_inds::NTuple{NR,Index};
               truncate::Int=100,
               lefttags::String="Link,u",
               righttags::String="Link,v",
               matrixtype::Type{T}=Hermitian) where {T,NL,NR}
  Lis = IndexSet(left_inds...)
  Ris = IndexSet(right_inds...)
  #TODO: make this a debug level check
  IndexSet(left_inds...,right_inds...) ≠ inds(A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,Lis...,Ris...)
  #TODO: More of the index analysis should be moved out of storage_eigen
  Uis,Ustore,Dis,Dstore = storage_eigen(store(A),Lis,Ris,matrixtype,truncate,lefttags,righttags)
  return ITensor(Uis,Ustore),ITensor(Dis,Dstore)
end

function eigen(A::ITensor,
               left_tags::NTuple{NL,String},
               right_tags::NTuple{NR,String};
               kwargs...) where {NL,NR}
  Linds = Index[] 
  Rinds = Index[]
  for tags ∈ left_tags
    push!(Linds,findinds(A,tags)...)
  end
  for tags ∈ right_tags
    push!(Rinds,findinds(A,tags)...)
  end
  return eigen(A,NTuple{length(Linds),Index}(Linds),NTuple{length(Rinds),Index}(Rinds);kwargs...)
end

function eigen(A::ITensor,
               left_tags::String,
               right_tags::String;
               kwargs...) where {NL,NR}
  return eigen(A,(left_tags,),(right_tags,);kwargs...)
end

function show_info(io::IO,
                   T::ITensor)
  print(io,"ITensor ord=$(order(T))")
  for i = 1:order(T)
    print(io," ",inds(T)[i])
  end
  print(io,"\n",typeof(store(T)))
end

function show(io::IO,T::ITensor)
  show_info(io,T)
  print(io,"\n")
  Base.print_array(io,reshape(data(store(T)),dims(T)))
end

function show(io::IO,
              mime::MIME"text/plain",
              T::ITensor)
  show_info(io,T)
end



