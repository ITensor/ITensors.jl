
struct ITensor
  inds::IndexSet
  store::TensorStorage
  #TODO: check that the storage is consistent with the
  #total dimension of the indices
  ITensor(is::IndexSet,st::TensorStorage) = new(is,st)
end

ITensor() = ITensor(IndexSet(),Dense{Nothing}())
ITensor(is::IndexSet) = ITensor(Float64,is...)
ITensor(inds::Index...) = ITensor(IndexSet(inds...))

function ITensor(::Type{T},
                 inds::IndexSet) where {T<:Number}
  return ITensor(inds,Dense{float(T)}(zero(float(T)),dim(inds)))
end
ITensor(::Type{T},inds::Index...) where {T<:Number} = ITensor(T,IndexSet(inds...))

function ITensor(::UndefInitializer,
                 inds::IndexSet)
  return ITensor(inds,Dense{Float64}(undef,dim(inds)))
end
ITensor(x::UndefInitializer,inds::Index...) = ITensor(x,IndexSet(inds...))

function ITensor(x::S,inds::IndexSet) where {S<:Number}
  return ITensor(inds,Dense{float(S)}(float(x),dim(inds)))
end
ITensor(x::S,inds::Index...) where {S<:Number} = ITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function ITensor(A::Array{S},inds::IndexSet) where {S<:Number}
  return ITensor(inds,Dense{float(S)}(float(vec(A))))
end
ITensor(A::Array{S},inds::Index...) where {S<:Number} = ITensor(A,IndexSet(inds...))

# Convert to complex
complex(T::ITensor) = ITensor(inds(T),storage_complex(store(T)))

inds(T::ITensor) = T.inds

# This constructor allows many IndexSet
# set operations to work with ITensors
IndexSet(T::ITensor) = inds(T)

store(T::ITensor) = T.store

eltype(T::ITensor) = eltype(store(T))

order(T::ITensor) = order(inds(T))

dims(T::ITensor) = dims(inds(T))

dim(T::ITensor) = dim(inds(T))

isNull(T::ITensor) = (store(T) isa Dense{Nothing})

copy(T::ITensor) = ITensor(copy(inds(T)),copy(store(T)))

copyto!(R::ITensor,T::ITensor) = (R = copy(T))

Array(T::ITensor) = storage_convert(Array,store(T),inds(T))

function getindex(T::ITensor,vals::Int...) 
  if order(T) ≠ length(vals) 
    error("In getindex(::ITensor,::Int..), number of \\
           values provided ($(length(vals))) must equal \\
           order of ITensor ($(order(T)))")
  end
  storage_getindex(store(T),inds(T),vals...)
end

function getindex(T::ITensor,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return getindex(T,vals...)
end

function getindex(T::ITensor,ivs::Union{IndexVal, AbstractVector{IndexVal}}...)
  p = calculate_permutation(inds(T),map(x->x isa IndexVal ? x : x[1], ivs))
  vals = map(x->x isa IndexVal ? val(x) : val.(x), ivs)
  storage_getindex(store(T),inds(T),vals...)
end

getindex(T::ITensor) = scalar(T)

setindex!(T::ITensor,x::Number,vals::Int...) = storage_setindex!(store(T),inds(T),x,vals...)

function setindex!(T::ITensor,x::Number,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return setindex!(T,x,vals...)
end

function setindex!(T::ITensor,
                   x::Union{<:Number, AbstractArray{<:Number}}, 
                   ivs::Union{IndexVal, AbstractVector{IndexVal}}...)
  remap_ivs = map(x->x isa IndexVal ? x : x[1], ivs)
  p = calculate_permutation(inds(T),remap_ivs)
  vals = map(x->x isa IndexVal ? val(x) : val.(x), ivs)
  storage_setindex!(store(T),inds(T),x,vals...)
end

function fill!(T::ITensor,
               x::Number)
  # TODO: automatically switch storage type if needed
  storage_fill!(store(T),x)
end

prime(A::ITensor,vargs...)= ITensor(prime(inds(A),vargs...),store(A))

adjoint(A::ITensor) = prime(A)

setprime(A::ITensor,vargs...) = ITensor(setprime(inds(A),vargs...),store(A))

noprime(A::ITensor,vargs...) = ITensor(noprime(inds(A),vargs...),store(A))

# TODO: remove in favor of replacetags(...)
mapprime(A::ITensor,vargs...) = ITensor(mapprime(inds(A),vargs...),store(A))

# TODO: remove in favor of swaptags(...)
swapprime(A::ITensor,vargs...) = ITensor(swapprime(inds(A),vargs...),store(A))

addtags(A::ITensor,vargs...) = ITensor(addtags(inds(A),vargs...),store(A))

removetags(A::ITensor,vargs...) = ITensor(removetags(inds(A),vargs...),store(A))

replacetags(A::ITensor,vargs...) = ITensor(replacetags(inds(A),vargs...),store(A))

swaptags(A::ITensor,vargs...) = ITensor(swaptags(inds(A),vargs...),store(A))

function ==(A::ITensor,B::ITensor)
  !hassameinds(A,B) && return false 
  p = calculate_permutation(inds(B),inds(A))
  for i ∈ CartesianIndices(dims(A))
    A[Tuple(i)...] ≠ B[Tuple(i)[p]...] && return false
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
    @show inds(T)
    error("ITensor is not a scalar")
  end
  return storage_scalar(store(T))
end

randn!(T::ITensor) = storage_randn!(store(T))

function randomITensor(::Type{S},inds) where {S<:Number}
  T = ITensor(S,IndexSet(inds))
  randn!(T)
  return T
end
randomITensor(::Type{S},inds::Index...) where {S<:Number} = randomITensor(S,IndexSet(inds...))
randomITensor(inds) = randomITensor(Float64,IndexSet(inds))
randomITensor(inds::Index...) = randomITensor(Float64,IndexSet(inds...))

norm(T::ITensor) = storage_norm(store(T))
dag(T::ITensor) = ITensor(storage_dag(store(T),inds(T))...)

function permute(T::ITensor,permTinds)
  permTis = IndexSet(permTinds)
  permTstore = typeof(store(T))(dim(T))
  storage_permute!(permTstore,permTis,store(T),inds(T))
  return ITensor(permTis,permTstore)
end
permute(T::ITensor,inds::Index...) = permute(T,IndexSet(inds...))

function add!(A::ITensor,B::ITensor)
  storage_add!(store(A),inds(A),store(B),inds(B))
end

function add!(A::ITensor,B::ITensor,x::Number)
  storage_add!(store(A),inds(A),store(B),inds(B),x)
end

function scale!(A::ITensor,x::Number)
  storage_mult!(store(A), x)  
end

function *(A::ITensor,x::Number)
    B = copy(A)
    storage_mult!(store(B), x)
    return B
end
*(x::Number,A::ITensor) = A*x
#TODO: make a proper element-wise division
/(A::ITensor,x::Number) = A*(1.0/x)

-(A::ITensor) = -one(eltype(A))*A
function +(A::ITensor,B::ITensor)
  A==B && return 2*A
  C = copy(A)
  add!(C,B)
  return C
end
-(A::ITensor,B::ITensor) = A+(-B)

#TODO: Add special case of A==B
#A==B && return ITensor(norm(A)^2)
#TODO: Add more of the contraction logic here?
#We can move the logic of getting the integer labels,
#etc. since they are generic for all storage types
function *(A::ITensor,B::ITensor)
  (Cis,Cstore) = storage_contract(store(A),inds(A),store(B),inds(B))
  C = ITensor(Cis,Cstore)
  return C
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
  if !isNull(T)
    Base.print_array(io,reshape(data(store(T)),dims(T)))
  end
end

function show(io::IO,
              mime::MIME"text/plain",
              T::ITensor)
  show_info(io,T)
end

function similar(T::ITensor,
                 element_type=eltype(T))::ITensor
  if element_type != eltype(T)
    error("similar(::ITensor) currently only defined for same element type")
  end
  return copy(T)
end

function mul!(R::ITensor,
              T::ITensor,
              fac::Number)
  R = copy(T)
  R *= fac
end

rmul!(T::ITensor,fac::Number) = (T *= fac)

dot(A::ITensor,B::ITensor) = scalar(dag(A)*B)

function axpy!(a::Number,
               v::ITensor,
               w::ITensor)
  w = a*v + w
end

function axpby!(a::Number,
                v::ITensor,
                b::Number,
                w::ITensor)
  w = a*v + b*w
end

#TODO: This is just a stand-in for a proper delta/diag storage type
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
