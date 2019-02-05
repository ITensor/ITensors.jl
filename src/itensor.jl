export ITensor,
       norm,
       delta,
       dims,
       δ,
       replaceindex!,
       inds,
       isNull,
       normalize!,
       multSiteOps,
       order,
       permute,
       randomITensor,
       scalar,
       store

mutable struct ITensor
  inds::IndexSet
  store::TensorStorage
  #TODO: check that the storage is consistent with the
  #total dimension of the indices (possibly only in debug mode);
  ITensor(is::IndexSet,st::T) where T = new(is,st)
end

ITensor() = ITensor(IndexSet(),Dense{Nothing,Vector{Nothing}}())
ITensor(is::IndexSet) = ITensor(Float64,is...)
ITensor(inds::Index...) = ITensor(IndexSet(inds...))

# ITensor c-tors
function ITensor(::Type{T},inds::IndexSet) where {T<:Number}
    return ITensor(inds,Dense{T, Vector{T}}(dim(inds)))
end
ITensor(::Type{T},inds::Index...) where {T<:Number} = ITensor(T,IndexSet(inds...))

function ITensor(::UndefInitializer,
                 inds::IndexSet)
  return ITensor(inds,Dense{Float64}(undef,dim(inds)))
end
ITensor(x::UndefInitializer,inds::Index...) = ITensor(x,IndexSet(inds...))

function ITensor(x::S,inds::IndexSet) where {S<:Number}
    return ITensor(inds,Dense{S, Vector{S}}(x,dim(inds)))
end
ITensor(x::S,inds::Index...) where {S<:Number} = ITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function ITensor(A::Array{S},inds::IndexSet) where {S<:Number}
  if length(A) ≠ dim(inds)
    error("In ITensor(Array,IndexSet), length of Array must match total dimension of IndexSet")
  end
  return ITensor(inds,Dense{S, typeof(A)}(A))
end
ITensor(A::Array{S},inds::Index...) where {S<:Number} = ITensor(A,IndexSet(inds...))

# Convert to complex
complex(T::ITensor) = ITensor(inds(T),storage_complex(store(T)))

inds(T::ITensor) = T.inds

# This constructor allows many IndexSet
# set operations to work with ITensors
IndexSet(T::ITensor) = inds(T)

isNull(T::ITensor) = (typeof(store(T)) == Dense{Nothing} || dims(T) == ())

copyto!(R::ITensor,T::ITensor) = (R = copy(T))

inds(T::ITensor) = T.inds
store(T::ITensor) = T.store
eltype(T::ITensor) = eltype(store(T))

order(T::ITensor) = order(inds(T))
dims(T::ITensor) = dims(inds(T))
dim(T::ITensor) = dim(inds(T))

copy(T::ITensor) = ITensor(copy(inds(T)),copy(store(T)))

Array(T::ITensor) = storage_convert(Array,store(T),inds(T))
CuArray(T::ITensor) = storage_convert(CuArray,store(T),inds(T))

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
  return storage_getindex(store(T),inds(T),vals...)
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
  return T
end

function fill!(T::ITensor,
               x::Number)
  # TODO: automatically switch storage type if needed
  storage_fill!(store(T),x)
  return T
end

function replaceindex!(A::ITensor,i::Index,j::Index)
  pos = indexpositions(A,i)
  inds(A)[pos[1]] = j
  return A
end

prime(A::ITensor,vargs...)= ITensor(prime(inds(A),vargs...),store(A))
prime!(A::ITensor,vargs...)= ( prime!(inds(A),vargs...); return A )

adjoint(A::ITensor) = prime(A)

setprime(A::ITensor,vargs...) = ITensor(setprime(inds(A),vargs...),store(A))

noprime(A::ITensor,vargs...) = ITensor(noprime(inds(A),vargs...),store(A))

mapprime(A::ITensor,vargs...) = ITensor(mapprime(inds(A),vargs...),store(A))

swapprime(A::ITensor,vargs...) = ITensor(swapprime(inds(A),vargs...),store(A))

addtags(A::ITensor,vargs...) = ITensor(addtags(inds(A),vargs...),store(A))

removetags(A::ITensor,vargs...) = ITensor(removetags(inds(A),vargs...),store(A))

replacetags(A::ITensor,vargs...) = ITensor(replacetags(inds(A),vargs...),store(A))

settags(A::ITensor,vargs...) = ITensor(settags(inds(A),vargs...),store(A))

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

function randn!(T::ITensor)
  storage_randn!(store(T))
  return T
end

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
permute(T::ITensor,new_inds::Index...) = permute(T,IndexSet(new_inds...))

add!(A::ITensor, B::ITensor) = storage_add!(store(A),inds(A),store(B),inds(B))

#TODO: improve these using a storage_mult call
*(A::ITensor,x::Number) = A*ITensor(x)
*(x::Number,A::ITensor) = A*x
function /(A::ITensor,x::Number)
    ITensor(inds(A), (1.0/x)*store(A))
end
-(A::ITensor) = -one(eltype(A))*A
function +(A::ITensor,B::ITensor)
  #A==B && return 2*A
  C = copy(A)
  add!(C,B)
  return C
end
function -(A::ITensor,B::ITensor)
  C = copy(A)
  add!(C,-1,B)
  return C
end

#TODO: Add special case of A==B
#A==B && return ITensor(norm(A)^2)
#TODO: Add more of the contraction logic here?
#We can move the logic of getting the integer labels,
#etc. since they are generic for all storage types
function *(A::ITensor,B::ITensor)
  #TODO: Add special case of A==B
  #A==B && return ITensor(norm(A)^2)
  (Cis,Cstore) = storage_contract(store(A),inds(A),store(B),inds(B))
  C = ITensor(Cis,Cstore)
  return C
end

dot(A::ITensor,B::ITensor) = scalar(dag(A)*B)

#######################################################################
#
# In-place operations
#

"""
normalize!(T::ITensor)

Normalize an ITensor in-place, such that norm(T)==1.
"""
normalize!(T::ITensor) = scale!(T,1/norm(T))

"""
copyto!(B::ITensor, A::ITensor)

Copy the contents of ITensor A into ITensor B.

B .= A
"""
function copyto!(A::ITensor,B::ITensor)
  storage_copyto!(store(A),inds(A),store(B),inds(B))
  return A
end

"""
add!(B::ITensor, A::ITensor)

Add ITensors B and A and store the result in B.

B .+= A
"""
function add!(B::ITensor,A::ITensor)
  B.store = storage_add!(store(B),inds(B),store(A),inds(A))
end

"""
add!(B::ITensor,α::Number,A::ITensor)

Add ITensors B and α*A and store the result in B.

B .+= α .* A
"""
function add!(A::ITensor,x::Number,B::ITensor)
  A.store = storage_add!(store(A),inds(A),store(B),inds(B),x)
  return A
end

"""
add!(C::ITensor,α::Number,A::ITensor,β::Number,B::ITensor)

Add ITensors α*A and β*B and store the result in C.

C .= α .* A .+ β .* B
"""
function add!(A::ITensor,y::Number,x::Number,B::ITensor)
  A.store = storage_add!(store(A),inds(A),y,store(B),inds(B),x)
  return A
end

"""
axpy!(a::Number,v::ITensor,w::ITensor)

w .+= a .* v
"""
axpy!(a::Number,v::ITensor,w::ITensor) = add!(w,a,v)

# This is not implemented correctly
#"""
#w .= a .* v + b .* w
#"""
#axpby!(a::Number,v::ITensor,b::Number,w::ITensor) = add!(w,b,w,a,v)

"""
scale!(A::ITensor,x::Number)

Scale the ITensor A by x in-place.

A .*= x
"""
function scale!(A::ITensor,x::Number)
  storage_mult!(store(A), x)
  return A
end

"""
mul!(A::ITensor,x::Number,B::ITensor)

Multiply ITensor B with x and store the result in A.

A .*= x .* B
"""
function mul!(A::ITensor,x::Number,B::ITensor)
  storage_copyto!(store(A),inds(A),store(B),inds(B),x)
  return A
end
mul!(R::ITensor,T::ITensor,fac::Number) = mul!(R,fac,T)

rmul!(T::ITensor,fac::Number) = scale!(T,fac)

#TODO: This is just a stand-in for a proper delta/diag storage type
"""
delta(::Type{T},inds::Index...)

Make a diagonal ITensor with all diagonal elements 1.

WARNING: This is just a stand-in for a proper delta/diag storage type
"""
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

function multSiteOps(A::ITensor,
                     B::ITensor)::ITensor
  R = copy(A)
  prime!(R,"Site")
  R *= B
  return mapprime(R,2,1)
end

