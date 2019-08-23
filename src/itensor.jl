export ITensor,
       norm,
       combiner,
       combinedindex,
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
       diagITensor,
       scalar,
       store,
       dense


mutable struct ITensor
  inds::IndexSet
  store::TensorStorage
  #TODO: check that the storage is consistent with the
  #total dimension of the indices (possibly only in debug mode);
  ITensor(is::IndexSet,st::T) where T = new(is,st)
end

#
# Dense ITensor constructors
#

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
  return ITensor(inds,Dense{Float64}(Vector{Float64}(undef,dim(inds))))
end
ITensor(x::UndefInitializer,inds::Index...) = ITensor(x,IndexSet(inds...))

function ITensor(x::S,inds::IndexSet) where {S<:Number}
  return ITensor(inds,Dense{float(S)}(float(x),dim(inds)))
end
ITensor(x::S,inds::Index...) where {S<:Number} = ITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function ITensor(A::Array{S},inds::IndexSet) where {S<:Number}
    length(A) ≠ dim(inds) && throw(DimensionMismatch("In ITensor(Array,IndexSet), length of Array ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))"))
  return ITensor(inds,Dense{float(S)}(float(vec(A))))
end
ITensor(A::Array{S},inds::Index...) where {S<:Number} = ITensor(A,IndexSet(inds...))

#
# Diag ITensor constructors
#

"""
diagITensor(::Type{T}, is::IndexSet)

Make a sparse ITensor of element type T with non-zero elements 
only along the diagonal. Defaults to having `zero(T)` along the diagonal.
The storage will have Diag type.
"""
function diagITensor(::Type{T},
                     is::IndexSet) where {T<:Number}
  return ITensor(is,Diag{Vector{T}}(zero(T),minDim(is)))
end

"""
diagITensor(::Type{T}, is::Index...)

Make a sparse ITensor of element type T with non-zero elements 
only along the diagonal. Defaults to having `zero(T)` along the diagonal.
The storage will have Diag type.
"""
diagITensor(::Type{T},inds::Index...) where {T<:Number} = diagITensor(T,IndexSet(inds...))

"""
diagITensor(v::Vector{T}, is::IndexSet)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the values stored in `v` and 
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(v::Vector{T},
                     is::IndexSet) where {T<:Number}
  length(v) ≠ minDim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
  return ITensor(is,Diag{Vector{float(T)}}(v))
end

"""
diagITensor(v::Vector{T}, is::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the values stored in `v` and 
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(v::Vector{T},
                     is::Index...) where {T<:Number}
  return diagITensor(v,IndexSet(is...))
end

"""
diagITensor(is::IndexSet)

Make a sparse ITensor of element type Float64 with non-zero elements 
only along the diagonal. Defaults to storing zeros along the diagonal.
The storage will have Diag type.
"""
diagITensor(is::IndexSet) = ITensor(is,Diag{Vector{Float64}}(zero(Float64),minDim(is)))

"""
diagITensor(is::Index...)

Make a sparse ITensor of element type Float64 with non-zero elements 
only along the diagonal. Defaults to storing zeros along the diagonal.
The storage will have Diag type.
"""
diagITensor(inds::Index...) = diagITensor(IndexSet(inds...))

"""
diagITensor(x::T, is::IndexSet) where {T<:Number}

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `x` and
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(x::T,
                     is::IndexSet) where {T<:Number}
  return ITensor(is,Diag{Vector{float(T)}}(float(x),minDim(is)))
end

"""
diagITensor(x::T, is::Index...) where {T<:Number}

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `x` and
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(x::T,
                     is::Index...) where {T<:Number}
  return diagITensor(x,IndexSet(is...))
end

"""
    delta(::Type{T},inds::IndexSet)

Make a diagonal ITensor with all diagonal elements 1.
"""
function delta(::Type{T},is::IndexSet) where {T}
  return ITensor(is,Diag{float(T)}(one(T)))
end

"""
    delta(::Type{T},inds::Index...)

Make a diagonal ITensor with all diagonal elements 1.
"""
function delta(::Type{T},is::Index...) where {T}
  return delta(T,IndexSet(is...))
end

delta(is::IndexSet) = delta(Float64,is)

delta(is::Index...) = delta(IndexSet(is...))
const δ = delta

"""
dense(T::ITensor)

Make a copy of the ITensor where the storage is the dense version.
For example, an ITensor with Diag storage will become Dense storage.
"""
dense(T::ITensor) = ITensor(inds(T),storage_dense(store(T),inds(T)))

"""
complex(T::ITensor)

Convert to the complex version of the storage.
"""
Base.complex(T::ITensor) = ITensor(inds(T),storage_complex(store(T)))

inds(T::ITensor) = T.inds
store(T::ITensor) = T.store

# This constructor allows many IndexSet
# set operations to work with ITensors
IndexSet(T::ITensor) = inds(T)

eltype(T::ITensor) = eltype(store(T))

"""
    order(A::ITensor) = ndims(A)

The number of indices, `length(inds(A))`.
"""
order(T::ITensor) = order(inds(T))
ndims(T::ITensor) = order(inds(T))

"""
    dim(A::ITensor) = length(A)

The total number of entries, `prod(size(A))`.
"""
dim(T::ITensor) = dim(inds(T))

"""
    dims(A::ITensor) = size(A)

Tuple containing `size(A,d) == dim(inds(A)[d]) for d in 1:ndims(A)`.
"""
dims(T::ITensor) = dims(inds(T))
size(A::ITensor) = dims(inds(A))
size(A::ITensor, d::Int) = d in 1:ndims(A) ? dim(inds(A)[d]) :
  d>0 ? 1 : error("arraysize: dimension out of range")

isNull(T::ITensor) = (store(T) isa Dense{Nothing})

copy(T::ITensor) = ITensor(copy(inds(T)),copy(store(T)))

Array(T::ITensor) = storage_convert(Array,store(T),inds(T))

Array(T::ITensor,ninds::Index...) = storage_convert(Array,store(T),inds(T),IndexSet(ninds))

function Matrix(A::ITensor,i1::Index,i2::Index)  
  if ndims(A) != 2
    throw(DimensionMismatch("Matrix() expected a 2-index ITensor"))
  end
  return Array(A,i1,i2)
end

Matrix(A::ITensor) = Matrix(A,inds(A)...)

function Vector(A::ITensor)
  if ndims(A) != 1
    throw(DimensionMismatch("Vector() expected a 1-index ITensor"))
  end
  return Array(A,inds(A)...)
end

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
  vals = map(x->x isa IndexVal ? val(x) : val.(x), ivs[p])
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
  vals = map(x->x isa IndexVal ? val(x) : val.(x), ivs[p])
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

replacetags!(A::ITensor,vargs...) = replacetags!(A.inds,vargs...)

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
  !(order(T)==0 || dim(T)==1) && throw(ArgumentError("ITensor with inds $(inds(T)) is not a scalar"))
  return storage_scalar(store(T))
end

function randn!(T::ITensor)
  storage_randn!(store(T))
  return T
end

const Indices = Union{IndexSet,Tuple{Vararg{Index}}}

function randomITensor(::Type{S},inds::Indices) where {S<:Number}
  T = ITensor(S,IndexSet(inds))
  randn!(T)
  return T
end
randomITensor(::Type{S},inds::Index...) where {S<:Number} = randomITensor(S,IndexSet(inds...))
randomITensor(inds::Indices) = randomITensor(Float64,IndexSet(inds))
randomITensor(inds::Index...) = randomITensor(Float64,IndexSet(inds...))

function combiner(inds::IndexSet; kwargs...)
    tags = get(kwargs, :tags, "CMB,Link")
    new_ind = Index(prod(dims(inds)), tags)
    new_is = IndexSet(new_ind, inds)
    return ITensor(new_is, CombinerStorage(new_ind))
end
combiner(inds::Index...; kwargs...) = combiner(IndexSet(inds...); kwargs...)

combinedindex(T::ITensor) = store(T) isa CombinerStorage ? store(T).ci : Index()

norm(T::ITensor) = storage_norm(store(T))
dag(T::ITensor) = ITensor(storage_dag(store(T),inds(T))...)

function permute(T::ITensor,permTinds)
  permTis = IndexSet(permTinds)
  permTstore = typeof(store(T))(dim(T))
  storage_permute!(permTstore,permTis,store(T),inds(T))
  return ITensor(permTis,permTstore)
end
permute(T::ITensor,inds::Index...) = permute(T,IndexSet(inds...))

function *(A::ITensor,x::Number)
    storeB = storage_mult(store(A), x)
    return ITensor(inds(A),storeB)
end
*(x::Number,A::ITensor) = A*x
#TODO: make a proper element-wise division
/(A::ITensor,x::Number) = A*(1.0/x)

-(A::ITensor) = -one(eltype(A))*A
function +(A::ITensor,B::ITensor)
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
  (Cis,Cstore) = storage_contract(store(A),inds(A),store(B),inds(B))
  C = ITensor(Cis,Cstore)
  if warnTensorOrder > 0 && order(C) >= warnTensorOrder
    println("Warning: contraction resulted in ITensor with $(order(C)) indices")
  end
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
```
B .= A
```
"""
function copyto!(A::ITensor,B::ITensor)
  storage_copyto!(store(A),inds(A),store(B),inds(B))
  return A
end

"""
    add!(B::ITensor, A::ITensor)
    add!(B::ITensor, α::Number, A::ITensor)

Add ITensors B and A (or α*A) and store the result in B.
```
B .+= A
B .+= α .* A
```
"""
function add!(B::ITensor,A::ITensor)
  # TODO: is replacing the storage entirely the best way
  # to do this logic? Is this worse in the case that
  # the storage type stays the same?
  B.store = storage_add!(store(B),inds(B),store(A),inds(A))
  return B
end

function add!(A::ITensor,x::Number,B::ITensor)
  # TODO: is replacing the storage entirely the best way
  # to do this logic? Is this worse in the case that
  # the storage type stays the same?
  A.store = storage_add!(store(A),inds(A),store(B),inds(B),x)
  return A
end

"""
    add!(A::ITensor, α::Number, β::Number, B::ITensor)

Add ITensors α*A and β*B and store the result in A.
```
A .= α .* A .+ β .* B
```
"""
function add!(A::ITensor,y::Number,x::Number,B::ITensor)
  # TODO: is replacing the storage entirely the best way
  # to do this logic? Is this worse in the case that
  # the storage type stays the same?
  A.store = storage_add!(y*store(A),inds(A),store(B),inds(B),x)
  return A
end

"""
    axpy!(a::Number, v::ITensor, w::ITensor)
```
w .+= a .* v
```
"""
axpy!(a::Number,v::ITensor,w::ITensor) = add!(w,a,v)

# This is not implemented correctly
#"""
#w .= a .* v + b .* w
#"""
#axpby!(a::Number,v::ITensor,b::Number,w::ITensor) = add!(w,b,w,a,v)

"""
    scale!(A::ITensor,x::Number) = rmul!(A,x)

Scale the ITensor A by x in-place. May also be written `rmul!`.
```
A .*= x
```
"""
function scale!(A::ITensor,x::Number)
  storage_mult!(store(A), x)
  return A
end

"""
    mul!(A::ITensor,x::Number,B::ITensor)

Scalar multiplication of ITensor B with x, and store the result in A.
Like `A .= x .* B`, and equivalent to `add!(A, 0, x, B)`.
"""
function mul!(A::ITensor,x::Number,B::ITensor)
  storage_copyto!(store(A),inds(A),store(B),inds(B),x)
  return A
end
mul!(R::ITensor,T::ITensor,fac::Number) = mul!(R,fac,T)

rmul!(T::ITensor,fac::Number) = scale!(T,fac)

function summary(io::IO,
                   T::ITensor)
  print(io,"ITensor ord=$(order(T))")
  for i = 1:order(T)
    print(io," ",inds(T)[i])
  end
  print(io," \n",typeof(store(T)))
end

# TODO: make a specialized printing from Diag
# that emphasizes the missing elements
function show(io::IO,T::ITensor)
  summary(io,T)
  print(io,"\n")
  if !isNull(T)
    Base.print_array(io,Array(T))
  end
end

function show(io::IO,
              mime::MIME"text/plain",
              T::ITensor)
  summary(io,T)
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

