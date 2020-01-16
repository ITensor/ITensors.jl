export ITensor,
       axpy!,
       combiner,
       combinedindex,
       delta,
       δ,
       exphermitian,
       replaceindex!,
       inds,
       isnull,
       scale!,
       matmul,
       mul!,
       order,
       permute,
       randomITensor,
       rmul!,
       diagITensor,
       dot,
       tensor,
       array,
       matrix,
       vector,
       norm,
       normalize!,
       scalar,
       set_warnorder,
       store,
       dense,
       real_if_close

"""
An ITensor is a tensor whose interface is 
independent of its memory layout. Therefore
it is not necessary to know the ordering
of an ITensor's indices, only which indices
an ITensor has. Operations like contraction
and addition of ITensors automatically
handle any memory permutations.
"""
mutable struct ITensor{N}
  store::TensorStorage
  inds::IndexSet{N}
  #TODO: check that the storage is consistent with the
  #total dimension of the indices (possibly only in debug mode);
  ITensor{N}(st,is::IndexSet{N}) where {N} = new{N}(st,is)
end
ITensor(st,is::IndexSet{N}) where {N} = ITensor{N}(st,is)

Tensors.inds(T::ITensor) = T.inds
Tensors.store(T::ITensor) = T.store

# TODO: do we need these? I think yes, for add!(::ITensor,::ITensor)
setinds!(T::ITensor,is...) = (T.inds = IndexSet(is...))
setstore!(T::ITensor,st::TensorStorage) = (T.store = st)

#
# Iteration over ITensors
#

"""
    CartesianIndices(A::ITensor)

Iterate over the CartesianIndices of an ITensor.
"""
Base.CartesianIndices(A::ITensor) = CartesianIndices(dims(A))

#
# Dense ITensor constructors
#

ITensor(T::Tensor{<:Number,N}) where {N} = ITensor{N}(store(T),inds(T))
ITensor{N}(T::Tensor{<:Number,N}) where {N} = ITensor{N}(store(T),inds(T))

# Convert the ITensor to a Tensor that shares the same
# data and indices as the ITensor
# TODO: should we define a `convert(::Type{<:Tensor},::ITensor)`
# method?
tensor(A::ITensor) = Tensor(store(A),inds(A))

"""
    ITensor(iset::IndexSet)

Construct an ITensor having indices
given by the IndexSet `iset`
"""
ITensor(is::IndexSet) = ITensor(Float64,is)
ITensor() = ITensor{0}(Dense{Nothing}(),IndexSet())
ITensor(inds::Vararg{Index,N}) where {N} = ITensor(IndexSet{N}(inds...))

function ITensor(::Type{T},
                 inds::IndexSet{N}) where {T<:Number,N}
  return ITensor{N}(Dense{float(T)}(zeros(float(T),dim(inds))),inds)
end
ITensor(::Type{T},inds::Index...) where {T<:Number} = ITensor(T,IndexSet(inds...))

function ITensor(::UndefInitializer,
                 inds::IndexSet{N}) where {N}
  return ITensor{N}(Dense{Float64}(Vector{Float64}(undef,dim(inds))),inds)
end
ITensor(x::UndefInitializer,inds::Index...) = ITensor(x,IndexSet(inds...))

function ITensor(x::S,inds::IndexSet{N}) where {S<:Number,N}
  return ITensor{N}(Dense{float(S)}(fill(float(x),dim(inds))),inds)
end

"""
    ITensor(x)

Construct a scalar ITensor with value `x`.

    ITensor(x,i,j,...)

Construct an ITensor with indices `i`,`j`,...
and all elements set to `x`.
"""
ITensor(x::S,inds::Index...) where {S<:Number} = ITensor(x,IndexSet(inds...))

function ITensor(A::Array{S},inds::IndexSet{N}) where {S<:Number,N}
  length(A) ≠ dim(inds) && throw(DimensionMismatch("In ITensor(Array,IndexSet), length of Array ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))"))
  return ITensor{N}(Dense{float(S)}(float(vec(A))),inds)
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
                     is::IndexSet{N}) where {T<:Number,N}
  return ITensor{N}(Diag(zeros(T,mindim(is))),is)
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
  length(v) ≠ mindim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
  return ITensor(Diag(float(v)),is)
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
diagITensor(is::IndexSet) = diagITensor(Float64,is)

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
  return ITensor(Diag(fill(float(x),mindim(is))),is)
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
  return ITensor(Diag(one(T)),is)
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
function Tensors.dense(T::ITensor)
  ITensor(dense(Tensor(store(T),inds(T))))
end

"""
complex(T::ITensor)

Convert to the complex version of the storage.
"""
Base.complex(T::ITensor) = ITensor(complex(Tensor(store(T),inds(T))))

# This constructor allows many IndexSet
# set operations to work with ITensors
IndexSet(T::ITensor) = inds(T)

Base.eltype(T::ITensor) = eltype(tensor(T))

"""
    order(A::ITensor) = ndims(A)

The number of indices, `length(inds(A))`.
"""
order(T::ITensor) = order(inds(T))
Base.ndims(T::ITensor) = order(inds(T))

"""
    dim(A::ITensor) = length(A)

The total number of entries, `prod(size(A))`.
"""
Tensors.dim(T::ITensor) = dim(inds(T))

"""
    dims(A::ITensor) = size(A)

Tuple containing `size(A,d) == dim(inds(A)[d]) for d in 1:ndims(A)`.
"""
Tensors.dims(T::ITensor) = dims(inds(T))
Base.size(A::ITensor) = dims(inds(A))
Base.size(A::ITensor{N}, d::Int) where {N} = d in 1:N ? dim(inds(A)[d]) :
  d>0 ? 1 : error("arraysize: dimension out of range")

isnull(T::ITensor) = (eltype(T) === Nothing)

Base.copy(T::ITensor{N}) where {N} = ITensor{N}(copy(tensor(T)))

"""
    Array{ElT}(T::ITensor, i:Index...)

Given an ITensor `T` with indices `i...`, returns
an Array with a copy of the ITensor's elements. The
order in which the indices are provided indicates
the order of the data in the resulting Array.
"""
function Base.Array{ElT,N}(T::ITensor{N},is::Vararg{Index,N}) where {ElT,N}
  return Array{ElT,N}(tensor(permute(T,is...)))::Array{ElT,N}
end

function Base.Array{ElT}(T::ITensor{N},is::Vararg{Index,N}) where {ElT,N}
  return Array{ElT,N}(T,is...)
end

function Base.Array(T::ITensor{N},is::Vararg{Index,N}) where {N}
  return Array{eltype(T),N}(T,is...)::Array{<:Number,N}
end

"""
    Matrix(T::ITensor, row_i:Index, col_i::Index)

Given an ITensor `T` with two indices `row_i` and `col_i`, returns
a Matrix with a copy of the ITensor's elements. The
order in which the indices are provided indicates
which Index is to be treated as the row index of the 
Matrix versus the column index.

"""
function Base.Matrix(T::ITensor{2},row_i::Index,col_i::Index)
  return Array(T,row_i,col_i)
end

function Base.Vector(T::ITensor{1},i::Index)
  return Array(T,i)
end

function Base.Vector{ElT}(T::ITensor{1}) where {ElT}
  return Vector{ElT}(T,inds(T)...)
end

function Base.Vector(T::ITensor{1})
  return Vector(T,inds(T)...)
end

scalar(T::ITensor) = T[]::Number

Base.getindex(T::ITensor{N},vals::Vararg{Int,N}) where {N} = tensor(T)[vals...]::Number

# Version accepting CartesianIndex, useful when iterating over
# CartesianIndices
Base.getindex(T::ITensor{N},I::CartesianIndex{N}) where {N} = tensor(T)[I]::Number

function Base.getindex(T::ITensor{N},
                       ivs::Vararg{IndexVal,N}) where {N}
  p = getperm(inds(T),ivs)
  vals = permute(val.(ivs),p)
  return T[vals...]
end

# TODO: we should figure out if this is how we want to do
# slicing
#function getindex(T::ITensor,ivs::AbstractVector{IndexVal}...)
#  p = getperm(inds(T),map(x->x[1], ivs))
#  vals = map(x->val.(x), ivs[[p...]])
#  return Tensor(store(T),inds(T))[vals...]
#end

Base.getindex(T::ITensor) = tensor(T)[]

Base.setindex!(T::ITensor{N},x::Number,vals::Vararg{Int,N}) where {N} = (tensor(T)[vals...] = x)

function Base.setindex!(T::ITensor,x::Number,ivs::IndexVal...)
  p = getperm(inds(T),ivs)
  vals = permute(val.(ivs),p)
  return T[vals...] = x
end

function Base.fill!(T::ITensor,
                    x::Number)
  # TODO: automatically switch storage type if needed?
  fill!(tensor(T),x)
  return T
end

# TODO: implement in terms of delta tensors (better for QNs)
function replaceindex!(A::ITensor,i::Index,j::Index)
  pos = indexpositions(A,i)
  inds(A)[pos[1]] = j
  return A
end

function replaceinds!(A::ITensor,inds1,inds2)
  is1 = IndexSet(inds1)
  is2 = IndexSet(inds2)
  pos = indexpositions(A,is1)
  for (j,p) ∈ enumerate(pos)
    inds(A)[p] = is2[j]
  end
  return A
end

# TODO: can we turn these into a macro?
# for f ∈ (prime,setprime,noprime,...)
#   f(A::ITensor,vargs...) = ITensor(store(A),f(inds(A),vargs...))
#   f!(A::ITensor,vargs...) = ( f!(inds(A),vargs...); return A )
# end

# TODO: implement more in-place versions

prime!(A::ITensor,vargs...)= ( prime!(inds(A),vargs...); return A )
prime(A::ITensor,vargs...)= ITensor(store(A),prime(inds(A),vargs...))
Base.adjoint(A::ITensor) = prime(A)

setprime(A::ITensor,vargs...) = ITensor(store(A),setprime(inds(A),vargs...))

noprime(A::ITensor,vargs...) = ITensor(store(A),noprime(inds(A),vargs...))

mapprime(A::ITensor,vargs...) = ITensor(store(A),mapprime(inds(A),vargs...))

swapprime(A::ITensor,vargs...) = ITensor(store(A),swapprime(inds(A),vargs...))

addtags(A::ITensor,vargs...) = ITensor(store(A),addtags(inds(A),vargs...))

removetags(A::ITensor,vargs...) = ITensor(store(A),removetags(inds(A),vargs...))

replacetags!(A::ITensor,vargs...) = ( replacetags!(inds(A),vargs...); return A )
replacetags(A::ITensor,vargs...) = ITensor(store(A),replacetags(inds(A),vargs...))

settags!(A::ITensor,vargs...) = ( settags!(inds(A),vargs...); return A )
settags(A::ITensor,vargs...) = ITensor(store(A),settags(inds(A),vargs...))

swaptags(A::ITensor,vargs...) = ITensor(store(A),swaptags(inds(A),vargs...))

# TODO: implement in a better way (more generically for other storage)
Base.:(==)(A::ITensor,B::ITensor) = (norm(A-B) == zero(promote_type(eltype(A),eltype(B))))

function Base.isapprox(A::ITensor,
                       B::ITensor;
                       atol::Real=0.0,
                       rtol::Real=Base.rtoldefault(eltype(A),eltype(B),atol))
    return norm(A-B) <= atol + rtol*max(norm(A),norm(B))
end

# TODO: bring this back or just use T[]?
# I think T[] may not be generic, since it may only work if order(T)==0
# so it may be nice to have a seperate scalar(T) for when dim(T)==1
#function scalar(T::ITensor)
#  !(order(T)==0 || dim(T)==1) && throw(ArgumentError("ITensor with inds $(inds(T)) is not a scalar"))
#  return scalar(tensor(store(T),inds(T)))
#end

function Random.randn!(T::ITensor)
  return randn!(tensor(T))
end

const Indices = Union{IndexSet,Tuple{Vararg{Index}}}

"""
    randomITensor([S,] inds)

Construct an ITensor with type S (default Float64) and indices inds, whose elements are normally distributed random numbers.
"""
function randomITensor(::Type{S},
                       inds::Indices) where {S<:Number}
  T = ITensor(S,IndexSet(inds))
  randn!(T)
  return T
end
function randomITensor(::Type{S},
                       inds::Index...) where {S<:Number}
  return randomITensor(S,IndexSet(inds...))
end
randomITensor(inds::Indices) = randomITensor(Float64,
                                             IndexSet(inds))
randomITensor(inds::Index...) = randomITensor(Float64,
                                              IndexSet(inds...))

function combiner(inds::IndexSet; kwargs...)
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = Index(prod(dims(inds)), tags)
  new_is = IndexSet(new_ind, inds)
  return ITensor(Combiner(),new_is),new_ind
end
combiner(inds::Index...; kwargs...) = combiner(IndexSet(inds...); kwargs...)

combinedindex(T::ITensor) = store(T) isa Combiner ? store(T).ci : nothing

LinearAlgebra.norm(T::ITensor) = norm(tensor(T))

function dag(T::ITensor)
  TT = conj(tensor(T))
  return ITensor(store(TT),dag(inds(T)))
end

function Tensors.permute(T::ITensor{N},new_inds) where {N}
  perm = getperm(new_inds,inds(T))
  Tp = permutedims(tensor(T),perm)
  return ITensor(Tp)::ITensor{N}
end
Tensors.permute(T::ITensor,inds::Index...) = permute(T,IndexSet(inds...))

function Base.:*(T::ITensor,x::Number)
  return ITensor(x*tensor(T))
end
Base.:*(x::Number,T::ITensor) = T*x
#TODO: make a proper element-wise division
Base.:/(A::ITensor,x::Number) = A*(1.0/x)

Base.:-(A::ITensor) = ITensor(-tensor(A))
function Base.:+(A::ITensor,B::ITensor)
  C = copy(A)
  C = add!(C,B)
  return C
end
function Base.:-(A::ITensor,B::ITensor)
  C = copy(A)
  C = add!(C,-1,B)
  return C
end

function Base.:*(A::ITensor,B::ITensor)
  (Alabels,Blabels) = compute_contraction_labels(inds(A),inds(B))
  CT = contract(tensor(A),Alabels,tensor(B),Blabels)
  C = ITensor(CT)
  warnTensorOrder = GLOBAL_PARAMS["WarnTensorOrder"]
  if warnTensorOrder > 0 && order(C) >= warnTensorOrder
    println("Warning: contraction resulted in ITensor with $(order(C)) indices")
  end
  return C
end

LinearAlgebra.dot(A::ITensor,B::ITensor) = (dag(A)*B)[]

"""
    exp(A::ITensor, Lis::IndexSet; hermitian = false)

Compute the exponential of the tensor `A` by treating it as a matrix ``A_{lr}`` with
the left index `l` running over all indices in `Lis` and `r` running over all
indices not in `Lis`. Must have `dim(Lis) == dim(inds(A))/dim(Lis)` for the exponentiation to
be defined.
When `ishermitian=true` the exponential of `Hermitian(A_{lr})` is
computed internally.
"""
function LinearAlgebra.exp(A::ITensor,
                           Linds,
                           Rinds = prime(IndexSet(Linds));
                           ishermitian = false)
  Lis,Ris = IndexSet(Linds),IndexSet(Rinds)
  Lpos,Rpos = getperms(inds(A),Lis,Ris)
  expAT = exp(tensor(A),Lpos,Rpos;ishermitian=ishermitian)
  return ITensor(expAT)
end

function exphermitian(A::ITensor,
                      Linds,
                      Rinds = prime(IndexSet(Linds))) 
  return exp(A,Linds,Rinds;ishermitian=true)
end

function matmul(A::ITensor,
                B::ITensor)
  R = mapprime(mapprime(A,1,2),0,1)
  R *= B
  return mapprime(R,2,1)
end

#######################################################################
#
# In-place operations
#

"""
    normalize!(T::ITensor)

Normalize an ITensor in-place, such that norm(T)==1.
"""
LinearAlgebra.normalize!(T::ITensor) = scale!(T,1/norm(T))

"""
    copyto!(B::ITensor, A::ITensor)

Copy the contents of ITensor A into ITensor B.
```
B .= A
```
"""
function Base.copyto!(R::ITensor{N},T::ITensor{N}) where {N}
  perm = getperm(inds(R),inds(T))
  TR = permutedims!(tensor(R),tensor(T),perm)
  return ITensor(TR)
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
add!(R::ITensor,T::ITensor) = add!(R,1,T)

function add!(R::ITensor{N},α::Number,T::ITensor{N}) where {N}
  return apply!(R,T,(r,t)->r+α*t)
end

"""
    add!(A::ITensor, α::Number, β::Number, B::ITensor)

Add ITensors α*A and β*B and store the result in A.
```
A .= α .* A .+ β .* B
```
"""
function add!(R::ITensor{N},αr::Number,αt::Number,T::ITensor{N}) where {N}
  return apply!(R,T,(r,t)->αr*r+αt*t)
end

function apply!(R::ITensor{N},T::ITensor{N},f::Function) where {N}
  perm = getperm(inds(R),inds(T))
  TR,TT = tensor(R),tensor(T)

  # TODO: Include type promotion from α
  TR = convert(promote_type(typeof(TR),typeof(TT)),TR)
  TR = permutedims!!(TR,TT,perm,f)

  setstore!(R,store(TR))
  setinds!(R,inds(TR))
  return R
end

"""
    axpy!(a::Number, v::ITensor, w::ITensor)
```
w .+= a .* v
```
"""
LinearAlgebra.axpy!(a::Number,v::ITensor,w::ITensor) = add!(w,a,v)

# This is not implemented correctly
#"""
#w .= a .* v + b .* w
#"""
#LinearAlgebra.axpby!(a::Number,v::ITensor,b::Number,w::ITensor) = add!(w,b,a,v)

"""
    scale!(A::ITensor,x::Number) = rmul!(A,x)

Scale the ITensor A by x in-place. May also be written `rmul!`.
```
A .*= x
```
"""
function Tensors.scale!(T::ITensor,x::Number)
  TT = tensor(T)
  scale!(TT,x)
  return T
end
LinearAlgebra.rmul!(T::ITensor,fac::Number) = scale!(T,fac)

"""
    mul!(A::ITensor,x::Number,B::ITensor)

Scalar multiplication of ITensor B with x, and store the result in A.
Like `A .= x .* B`.
"""
LinearAlgebra.mul!(R::ITensor,α::Number,T::ITensor) = apply!(R,T,(r,t)->α*t )
LinearAlgebra.mul!(R::ITensor,T::ITensor,α::Number) = mul!(R,α,T)

#######################################################################
#
# Developer functions
#

# TODO: make versions where the element type can be specified (for type
# inference).
Tensors.array(T::ITensor) = array(tensor(T))

"""
    matrix(T::ITensor)

Given an ITensor `T` with two indices, returns
a Matrix with a copy of the ITensor's elements,
or a view in the case the the ITensor's storage is Dense.
The ordering of the elements in the Matrix, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.
*Therefore this method is intended for developer use
only and not recommended for use in ITensor applications.*
"""
function Tensors.matrix(T::ITensor{2})
  return array(tensor(T))
end

function Tensors.vector(T::ITensor{1})
  return array(tensor(T))
end

#######################################################################
#
# Printing, reading and writing ITensors
#

function Base.summary(io::IO,
                      T::ITensor)
  print(io,"ITensor ord=$(order(T))")
  for i = 1:order(T)
    print(io," ",inds(T)[i])
  end
  print(io," \n",typeof(store(T)))
end

# TODO: make a specialized printing from Diag
# that emphasizes the missing elements
function Base.show(io::IO,T::ITensor)
  summary(io,T)
  print(io,"\n")
  if !isnull(T)
    Base.show(io,MIME"text/plain"(),tensor(T))
  end
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::ITensor)
  summary(io,T)
end

function Base.similar(T::ITensor,
                      element_type=eltype(T))
  return ITensor(similar(tensor(T),element_type))
end

function readcpp(io::IO,::Type{Dense{ValT}};kwargs...) where {ValT}
  format = get(kwargs,:format,"v3")
  if format=="v3"
    size = read(io,UInt64)
    data = Vector{ValT}(undef,size)
    for n=1:size
      data[n] = read(io,ValT)
    end
    return Dense(data)
  else
    throw(ArgumentError("read Dense: format=$format not supported"))
  end
end

function readcpp(io::IO,::Type{ITensor};kwargs...)
  format = get(kwargs,:format,"v3")
  if format=="v3"
    inds = readcpp(io,IndexSet;kwargs...)
    read(io,12) # ignore scale factor by reading 12 bytes
    storage_type = read(io,Int32)
    if storage_type==0 # Null
      store = Dense{Nothing}()
    elseif storage_type==1  # DenseReal
      store = readcpp(io,Dense{Float64};kwargs...)
    elseif storage_type==2  # DenseCplx
      store = readcpp(io,Dense{ComplexF64};kwargs...)
    elseif storage_type==3  # Combiner
      store = CombinerStorage(T.inds[1])
    #elseif storage_type==4  # DiagReal
    #elseif storage_type==5  # DiagCplx
    #elseif storage_type==6  # QDenseReal
    #elseif storage_type==7  # QDenseCplx
    #elseif storage_type==8  # QCombiner
    #elseif storage_type==9  # QDiagReal
    #elseif storage_type==10 # QDiagCplx
    #elseif storage_type==11 # ScalarReal
    #elseif storage_type==12 # ScalarCplx
    else
      throw(ErrorException("C++ ITensor storage type $storage_type not yet supported"))
    end
    return ITensor(store,inds)
  else
    throw(ArgumentError("read ITensor: format=$format not supported"))
  end
end

function set_warnorder(ord::Int)
  ITensors.GLOBAL_PARAMS["WarnTensorOrder"] = ord
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    T::ITensor)
  g = g_create(parent,name)
  attrs(g)["type"] = "ITensor"
  attrs(g)["version"] = 1
  write(g,"inds",inds(T))
  write(g,"store",store(T))
end

#function HDF5.read(parent::Union{HDF5File,HDF5Group},
#                   name::AbstractString)
#  g = g_open(parent,name)
#
#  try
#    typestr = read(attrs(g)["type"])
#    type_t = eval(Meta.parse(typestr))
#    res = read(parent,"name",type_t)
#    return res
#  end
#  return 
#end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{ITensor})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "ITensor"
    error("HDF5 group or file does not contain ITensor data")
  end
  inds = read(g,"inds",IndexSet)

  stypestr = read(attrs(g_open(g,"store"))["type"])
  stype = eval(Meta.parse(stypestr))

  store = read(g,"store",stype)

  return ITensor(store,inds)
end
