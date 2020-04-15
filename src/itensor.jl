
# This is explicitly imported
# so that it can be exported from
# ITensors (it is not exported from
# NDTensors)
import .NDTensors: dag

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
  """
  ITensor{N}(is::IndexSet{N}, st::TensorStorage)

  This is an internal constructor for an ITensor
  where the ITensor stores a view of the
  TensorStorage.
  """
  ITensor{N}(is, st::TensorStorage) where {N} = new{N}(st, is)
end

"""
itensor(st::TensorStorage, is)

Constructor for an ITensor from a TensorStorage
and a set of indices.
The ITensor stores a view of the TensorStorage.
"""
itensor(st::TensorStorage, is) = ITensor{length(is)}(is, st)

"""
ITensor(st::TensorStorage, is)

Constructor for an ITensor from a TensorStorage
and a set of indices.
The TensorStorage is copied (the ITensor
owns the storage data).
"""
ITensor(st::TensorStorage, is) = itensor(copy(st), is)

"""
inds(T::ITensor)

Return the indices of the ITensor as an IndexSet.
"""
NDTensors.inds(T::ITensor) = T.inds

"""
ind(T::ITensor, i::Int)

Get the Index of the ITensor along dimension i.
"""
NDTensors.ind(T::ITensor, i::Int) = inds(T)[i]

"""
store(T::ITensor)

Return a view of the TensorStorage of the ITensor.
"""
NDTensors.store(T::ITensor) = T.store

"""
data(T::ITensor)

Return a view of the raw data of the ITensor.

This is mostly an internal ITensor function, please
let the developers of ITensors.jl know if there is
functionality for ITensors that you would like
that is not currently available.
"""
data(T::ITensor) = NDTensors.data(store(T))

Base.similar(T::ITensor) = itensor(similar(tensor(T)))

Base.similar(T::ITensor,
             ::Type{ElT}) where {ElT<:Number} = itensor(similar(tensor(T),ElT))

setinds!(T::ITensor,is) = (T.inds = is; return T)

setstore!(T::ITensor,st::TensorStorage) = (T.store = st; return T)

setinds(T::ITensor, is) = itensor(store(T),is)

setstore(T::ITensor, st) = itensor(st,inds(T))

#
# Iteration over ITensors
#

"""
    CartesianIndices(A::ITensor)

Iterate over the CartesianIndices of an ITensor.
"""
Base.CartesianIndices(A::ITensor) = CartesianIndices(dims(A))

#
# ITensor constructors
#

"""
ITensor(::Tensor)

Convert a Tensor to an ITensor with a copy of the storage and
indices.

To make an ITensor that shares the same storage as the Tensor,
is the function `itensor(::Tensor)`.
"""
ITensor(T::Tensor{<:Number,N}) where {N} = ITensor{N}(store(T),
                                                      inds(T))

ITensor{N}(T::Tensor{<:Number,N}) where {N} = ITensor{N}(store(T),
                                                         inds(T))

"""
itensor(::Tensor)

Make an ITensor that shares the same storage as the Tensor and
has the same indices.
"""
itensor(T::Tensor) = itensor(store(T),
                             inds(T))

"""
tensor(::ITensor)

Convert the ITensor to a Tensor that shares the same
storage and indices as the ITensor.
"""
NDTensors.tensor(A::ITensor) = tensor(store(A),inds(A))

"""
    ITensor(iset::IndexSet)

Construct an ITensor having indices
given by the IndexSet `iset`
"""
ITensor(is::IndexSet) = ITensor(Float64,is)

ITensor(inds::Vararg{Index,N}) where {N} = ITensor(IndexSet(inds...))

# TODO: make this Dense(Float64[]), Dense([0.0]), Dense([1.0])?
ITensor() = itensor(Dense{Nothing}(),IndexSet())

function ITensor(::Type{ElT},
                 inds::IndexSet{N}) where {ElT<:Number,N}
  return itensor(Dense(ElT,dim(inds)),inds)
end

ITensor(::Type{ElT},
        inds::Index...) where {ElT<:Number} = ITensor(ElT,
                                                      IndexSet(inds...))

function ITensor(::Type{ElT},
                 ::UndefInitializer,
                 inds::IndexSet{N}) where {ElT<:Number,N}
  return itensor(Dense(ElT,undef,dim(inds)),inds)
end

ITensor(::Type{ElT},
        ::UndefInitializer,
        inds::Index...) where {ElT} = ITensor(ElT,
                                              undef,
                                              IndexSet(inds...))

function ITensor(::UndefInitializer,
                 inds::IndexSet{N}) where {N}
  return itensor(Dense(undef,dim(inds)),inds)
end
ITensor(::UndefInitializer,
        inds::Index...) = ITensor(undef,
                                  IndexSet(inds...))

function ITensor(x::Number,
                 inds::IndexSet)
  return itensor(Dense(float(x),dim(inds)),inds)
end

"""
    ITensor(x)

Construct a scalar ITensor with value `x`.

    ITensor(x,i,j,...)

Construct an ITensor with indices `i`,`j`,...
and all elements set to `float(x)`.

Note that the ITensor storage will be the closest
floating point version of the input value.
"""
ITensor(x::Number,
        inds::Index...) = ITensor(x,IndexSet(inds...))

"""
    itensor(A::Array, inds::IndexSet)
    itensor(A::Array, inds::Index...)

Construct an ITensor from an Array and a set if Indices,
where the ITensor stores a view of the Array data.
"""
function itensor(A::Array{<:Number},
                 inds::IndexSet)
  length(A) ≠ dim(inds) && throw(DimensionMismatch("In ITensor(Array,IndexSet), length of Array ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))"))
  return itensor(Dense(float(vec(A))),inds)
end

itensor(A::Array{<:Number},
        inds::Vararg{Index}) = itensor(A,IndexSet(inds...))

ITensor(A::Array, inds::IndexSet) = itensor(copy(A), inds)

#
# Diag ITensor constructors
#

"""
diagITensor(::Type{T}, is::IndexSet)

Make a sparse ITensor of element type T with non-zero elements 
only along the diagonal. Defaults to having `zero(T)` along the diagonal.
The storage will have Diag type.
"""
function diagITensor(::Type{ElT},
                     is::IndexSet) where {ElT}
  return itensor(Diag(ElT, mindim(is)), is)
end

"""
diagITensor(::Type{T}, is::Index...)

Make a sparse ITensor of element type T with non-zero elements 
only along the diagonal. Defaults to having `zero(T)` along the diagonal.
The storage will have Diag type.
"""
diagITensor(::Type{ElT},
            inds::Index...) where {ElT} = diagITensor(ElT,
                                                      IndexSet(inds...))

"""
diagITensor(v::Vector{T}, is::IndexSet)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the values stored in `v` and 
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(v::Vector{<:Number},
                     is::IndexSet)
  length(v) ≠ mindim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
  return itensor(Diag(float(v)),is)
end

"""
diagITensor(v::Vector{T}, is::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the values stored in `v` and 
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(v::Vector{<:Number},
                     is::Index...)
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
function diagITensor(x::Number,
                     is::IndexSet)
  return itensor(Diag(float(x), mindim(is)), is)
end

"""
diagITensor(x::T, is::Index...) where {T<:Number}

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `x` and
the ITensor will have element type `float(T)`.
The storage will have Diag type.
"""
function diagITensor(x::Number,
                     is::Index...)
  return diagITensor(x, IndexSet(is...))
end

"""
    delta(::Type{T},inds::IndexSet)

Make a diagonal ITensor with all diagonal elements 1.
"""
function delta(::Type{T},
               is::IndexSet) where {T<:Number}
  return itensor(Diag(one(T)), is)
end

"""
    delta(::Type{T},inds::Index...)

Make a diagonal ITensor with all diagonal elements 1.
"""
function delta(::Type{T},
               is::Index...) where {T<:Number}
  return delta(T, IndexSet(is...))
end

delta(is::IndexSet) = delta(Float64,is)

delta(is::Index...) = delta(IndexSet(is...))

const δ = delta

function setelt(iv::Union{IndexVal,PairIndexInt})
  A = ITensor(ind(iv))
  A[val(iv)] = 1.0
  return A
end

"""
dense(T::ITensor)

Make a new ITensor where the storage is the dense version,
avoiding allocating new data if possible.
For example, an ITensor with Diag storage will become Dense storage.
"""
NDTensors.dense(T::ITensor) = itensor(dense(tensor(T)))

"""
complex(T::ITensor)

Convert to the complex version of the storage.
"""
Base.complex(T::ITensor) = itensor(complex(tensor(T)))

Base.eltype(T::ITensor) = eltype(tensor(T))

"""
    order(A::ITensor) = ndims(A)

The number of indices, `length(inds(A))`.
"""
order(T::ITensor) = ndims(T)

Base.ndims(T::ITensor) = length(inds(T))

"""
    dim(A::ITensor) = length(A)

The total number of entries, `prod(size(A))`.
"""
NDTensors.dim(T::ITensor) = dim(inds(T))

"""
    dims(A::ITensor) = size(A)

Tuple containing `size(A,d) == dim(inds(A)[d]) for d in 1:ndims(A)`.
"""
NDTensors.dims(T::ITensor) = dims(inds(T))

Base.size(A::ITensor) = dims(inds(A))

Base.size(A::ITensor,
          d::Int) = dim(inds(A),d)

isnull(T::ITensor) = (eltype(T) === Nothing)

Base.copy(T::ITensor) = itensor(copy(tensor(T)))

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

Base.getindex(T::ITensor{N},
              vals::Vararg{Int,N}) where {N} = tensor(T)[vals...]::Number

# Version accepting CartesianIndex, useful when iterating over
# CartesianIndices
Base.getindex(T::ITensor{N},
              I::CartesianIndex{N}) where {N} = tensor(T)[I]::Number

function Base.getindex(T::ITensor, ivs...)
  p = NDTensors.getperm(inds(T), ivs)
  vals = NDTensors.permute(val.(ivs), p)
  return T[vals...]
end

Base.getindex(T::ITensor) = tensor(T)[]

function Base.setindex!(T::ITensor,x::Number,vals::Int...)
  fluxT = flux(T)
  (!isnothing(fluxT) && fluxT != flux(T,vals...)) && error("setindex! not consistent with current flux")
  tensor(T)[vals...] = x
  return T
end

function Base.setindex!(T::ITensor,x::Number,ivs...)
  p = NDTensors.getperm(inds(T),ivs)
  vals = NDTensors.permute(val.(ivs),p)
  T[vals...] = x
  return T
end

function Base.fill!(T::ITensor,
                    x::Number)
  # TODO: automatically switch storage type if needed?
  fill!(tensor(T),x)
  return T
end

itensor2inds(A::ITensor) = inds(A)
itensor2inds(A) = A


# in
hasind(A,i::Index) = i ∈ itensor2inds(A)

# issubset
hasinds(A,is) = is ⊆ itensor2inds(A)
hasinds(A,is::Index...) = hasinds(A,IndexSet(is...))

# issetequal
hassameinds(A,B) = issetequal(itensor2inds(A),
                              itensor2inds(B))

# intersect
commoninds(A...; kwargs...) = IndexSet(intersect(itensor2inds.(A)...;
                                                 kwargs...)...)

# firstintersect
commonind(A...; kwargs...) = firstintersect(itensor2inds.(A)...;
                                            kwargs...)

# symdiff
noncommoninds(A...; kwargs...) = IndexSet(symdiff(itensor2inds.(A)...;
                                               kwargs...)...)

# firstsymdiff
noncommonind(A...; kwargs...) = getfirst(symdiff(itensor2inds.(A)...;
                                                 kwargs...))

# setdiff
uniqueinds(A...; kwargs...) = IndexSet(setdiff(itensor2inds.(A)...;
                                               kwargs...)...)

# firstsetdiff
uniqueind(A...; kwargs...) = firstsetdiff(itensor2inds.(A)...;
                                          kwargs...)

# union
unioninds(A...; kwargs...) = IndexSet(union(itensor2inds.(A)...;
                                            kwargs...)...)

# firstsymdiff
unionind(A...; kwargs...) = getfirst(union(itensor2inds.(A)...;
                                           kwargs...))

firstind(A...; kwargs...) = getfirst(itensor2inds.(A)...;
                                     kwargs...)

NDTensors.inds(A...; kwargs...) = filter(itensor2inds.(A)...;
                                       kwargs...)

# in-place versions of priming and tagging
for fname in (:prime,
              :setprime,
              :noprime,
              :mapprime,
              :swapprime,
              :addtags,
              :removetags,
              :replacetags,
              :settags,
              :swaptags,
              :replaceind,
              :replaceinds)
  @eval begin
    $fname(f::Function,
           A::ITensor,
           args...) = setinds(A,$fname(f,
                                       inds(A),
                                       args...))

    $(Symbol(fname,:!))(f::Function,
                        A::ITensor,
                        args...) = setinds!(A,$fname(f,
                                                     inds(A),
                                                     args...))

    $fname(A::ITensor,
           args...;
           kwargs...) = setinds(A,$fname(inds(A),
                                         args...;
                                         kwargs...))

    $(Symbol(fname,:!))(A::ITensor,
                        args...;
                        kwargs...) = setinds!(A,$fname(inds(A),
                                                       args...;
                                                       kwargs...))
  end
end

"""
adjoint(A::ITensor)

For A' notation.
"""
Base.adjoint(A::ITensor) = prime(A)

function Base.:(==)(A::ITensor,B::ITensor)
  return norm(A-B) == zero(promote_type(eltype(A),eltype(B)))
end

function Base.isapprox(A::ITensor,
                       B::ITensor;
                       kwargs...)
    B = permute(dense(B), inds(A))
    return isapprox(array(A), array(B); kwargs...)
end

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

randomITensor(::Type{ElT}) where {ElT<:Number} = randomITensor(ElT,
                                                               IndexSet())

randomITensor() = randomITensor(Float64)

function combiner(inds::IndexSet;
                  kwargs...)
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = Index(prod(dims(inds)), tags)
  new_is = IndexSet(new_ind, inds...)
  return itensor(Combiner(),new_is),new_ind
end

combiner(inds::Index...;
         kwargs...) = combiner(IndexSet(inds...); kwargs...)
combiner(inds::Tuple{Vararg{Index}};
         kwargs...) = combiner(inds...; kwargs...)

# Special case when no indices are combined (useful for generic code)
function combiner(; kwargs...)
  return itensor(Combiner(),IndexSet()),nothing
end

combinedind(T::ITensor) = store(T) isa Combiner ? inds(T)[1] : nothing

LinearAlgebra.norm(T::ITensor) = norm(tensor(T))

function dag(T::ITensor; always_copy=false)
  TT = conj(tensor(T); always_copy=always_copy)
  return itensor(store(TT),dag(inds(T)))
end

"""
permute(T::ITensors, inds)

Return a new ITensor T with indices permuted according
to the input indices inds. The storage of the ITensor
is permuted accordingly.
"""
function permute(T::ITensor{N},
                 new_inds) where {N}
  perm = NDTensors.getperm(new_inds, inds(T))
  Tp = permutedims(tensor(T), perm)
  return itensor(Tp)::ITensor{N}
end

permute(T::ITensor,
        inds::Index...) = permute(T,
                                  IndexSet(inds...))

Base.:*(T::ITensor, x::Number) = itensor(x*tensor(T))

Base.:*(x::Number, T::ITensor) = T*x

#TODO: make a proper element-wise division
Base.:/(A::ITensor, x::Number) = A*(1.0/x)

Base.:-(A::ITensor) = itensor(-tensor(A))

function Base.:+(A::ITensor, B::ITensor)
  C = copy(A)
  C .+= B
  return C
end

function Base.:-(A::ITensor, B::ITensor)
  C = copy(A)
  C .-= B
  return C
end

"""
    *(A::ITensor, B::ITensor)

Contract ITensors A and B to obtain a new ITensor. This 
contraction `*` operator finds all matching indices common
to A and B and sums over them, such that the result will 
have only the unique indices of A and B. To prevent
indices from matching, their prime level or tags can be 
modified such that they no longer compare equal - for more
information see the documentation on Index objects.
"""
function Base.:*(A::ITensor, B::ITensor)
  (labelsA,labelsB) = compute_contraction_labels(inds(A),inds(B))
  CT = contract(tensor(A),labelsA,tensor(B),labelsB)
  C = itensor(CT)
  warnTensorOrder = GLOBAL_PARAMS["WarnTensorOrder"]
  if warnTensorOrder > 0 && order(C) >= warnTensorOrder
    @warn "Contraction resulted in ITensor with $(order(C)) indices"
  end
  return C
end

# TODO: define for contraction order optimization
#Base.:*(A1::ITensor,
#        A2::ITensor,
#        A3::ITensor, As::ITensor...)

function LinearAlgebra.mul!(C::ITensor, A::ITensor, B::ITensor,
                            α::Number, β::Number=0)
  (labelsC,labelsA,labelsB) = compute_contraction_labels(inds(C),
                                                         inds(A),
                                                         inds(B))
  CT = NDTensors.contract!!(tensor(C), labelsC,
                          tensor(A), labelsA,
                          tensor(B), labelsB,
                          α, β)
  C = itensor(CT)
  return C
end

# This is necessary for now since not all types implement contract!!
# with non-trivial α and β
function LinearAlgebra.mul!(C::ITensor, A::ITensor, B::ITensor)
  (labelsC,labelsA,labelsB) = compute_contraction_labels(inds(C),
                                                         inds(A),
                                                         inds(B))
  CT = NDTensors.contract!!(tensor(C), labelsC,
                          tensor(A), labelsA,
                          tensor(B), labelsB)
  C = itensor(CT)
  return C
end

# TODO: this will allow for contraction order optimization
#LinearAlgebra.mul!(R::ITensor,
#                   A1::ITensor,
#                   A2::ITensor,
#                   A3::ITensor, As...) = mul!(R, A1, *(A2, A3, As...))

LinearAlgebra.dot(A::ITensor, B::ITensor) = (dag(A)*B)[]

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
  Lpos,Rpos = NDTensors.getperms(inds(A), Lis, Ris)
  expAT = exp(tensor(A), Lpos, Rpos; ishermitian=ishermitian)
  return itensor(expAT)
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
LinearAlgebra.normalize!(T::ITensor) = (T .*= 1/norm(T))

"""
    copyto!(B::ITensor, A::ITensor)

Copy the contents of ITensor A into ITensor B.
```
B .= A
```
"""
function Base.copyto!(R::ITensor{N}, T::ITensor{N}) where {N}
  perm = NDTensors.getperm(inds(R), inds(T))
  TR = permutedims!(tensor(R), tensor(T), perm)
  return itensor(TR)
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
add!(R::ITensor, T::ITensor) = apply!(R,T,(r,t)->r+t)

add!(R::ITensor, α::Number, T::ITensor) = apply!(R,T,(r,t)->r+α*t)

add!(R::ITensor, T::ITensor, α::Number) = (R .+= α .* T)

"""
    add!(A::ITensor, α::Number, β::Number, B::ITensor)

Add ITensors α*A and β*B and store the result in A.
```
A .= α .* A .+ β .* B
```
"""
add!(R::ITensor,
     αr::Number,
     αt::Number,
     T::ITensor) = apply!(R,T,(r,t)->αr*r+αt*t)

function apply!(R::ITensor{N},
                T::ITensor{N},
                f::Function) where {N}
  perm = NDTensors.getperm(inds(R),inds(T))
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
LinearAlgebra.axpy!(a::Number,
                    v::ITensor,
                    w::ITensor) = (w .+= a .* v)

"""
axpby!(a,v,b,w)

```
w .= a .* v + b .* w
```
"""
LinearAlgebra.axpby!(a::Number,
                     v::ITensor,
                     b::Number,
                     w::ITensor) = (w .= a .* v + b .* w)

"""
    scale!(A::ITensor,x::Number) = rmul!(A,x)

Scale the ITensor A by x in-place. May also be written `rmul!`.
```
A .*= x
```
"""
function NDTensors.scale!(T::ITensor, x::Number)
  TT = tensor(T)
  scale!(TT,x)
  return T
end

LinearAlgebra.rmul!(T::ITensor, fac::Number) = scale!(T,fac)

"""
    mul!(A::ITensor,x::Number,B::ITensor)

Scalar multiplication of ITensor B with x, and store the result in A.
Like `A .= x .* B`.
"""
LinearAlgebra.mul!(R::ITensor,
                   α::Number,
                   T::ITensor) = apply!(R,T,(r,t)->α*t )

LinearAlgebra.mul!(R::ITensor,
                   T::ITensor,
                   α::Number) = (R .= α .* T)

#
# Block sparse related functions
# (Maybe create fallback definitions for dense tensors)
#

hasqns(T::ITensor) = hasqns(inds(T))

NDTensors.nnz(T::ITensor) = nnz(tensor(T))

NDTensors.nnzblocks(T::ITensor) = nnzblocks(tensor(T))

NDTensors.nzblock(T::ITensor, args...) = nzblock(tensor(T), args...)

NDTensors.nzblocks(T::ITensor) = nzblocks(tensor(T))

NDTensors.blockoffsets(T::ITensor) = blockoffsets(tensor(T))

flux(T::ITensor, args...) = flux(inds(T), args...)

function NDTensors.addblock!(T::ITensor,
                           args...)
  (!isnothing(flux(T)) && flux(T) ≠ flux(T, args...)) && 
   error("Block does not match current flux")
  addblock!(tensor(T), args...)
  return T
end

function flux(T::ITensor)
  !hasqns(T) && return nothing
  nnzblocks(T) == 0 && return nothing
  bofs = blockoffsets(T)
  block1 = nzblock(bofs, 1)
  return flux(T,block1)
end


#######################################################################
#
# Developer functions
#

# TODO: make versions where the element type can be specified (for type
# inference).
NDTensors.array(T::ITensor) = array(tensor(T))

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
function NDTensors.matrix(T::ITensor{2})
  return array(tensor(T))
end

function NDTensors.vector(T::ITensor{1})
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
    if hasqns(inds(T)[i])
      startstr = (i==1) ? "\n" : ""
      print(io,startstr,inds(T)[i])
    else
      print(io," ",inds(T)[i])
    end
  end
  print(io," \n",typeof(store(T)))
end

# TODO: make a specialized printing from Diag
# that emphasizes the missing elements
function Base.show(io::IO,
                   T::ITensor)
  #summary(io,T)
  println(io,"ITensor ord=$(order(T))")
  println(io)
  if !isnull(T)
    Base.show(io,MIME"text/plain"(),tensor(T))
  end
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::ITensor)
  summary(io,T)
end

function readcpp(io::IO,
                 ::Type{Dense{ValT}};
                 kwargs...) where {ValT}
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

function readcpp(io::IO,
                 ::Type{ITensor};
                 kwargs...)
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
    return itensor(store,inds)
  else
    throw(ArgumentError("read ITensor: format=$format not supported"))
  end
end

function setwarnorder!(ord::Int)
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

  return itensor(store,inds)
end

#
# Deprecations
#

@deprecate findindex(args...; kwargs...) firstind(args...; kwargs...)

@deprecate findinds(args...; kwargs...) inds(args...; kwargs...)

@deprecate commonindex(args...; kwargs...) commonind(args...; kwargs...)

@deprecate uniqueindex(args...; kwargs...) uniqueind(args...; kwargs...)

@deprecate replaceindex!(args...; kwargs...) replaceind!(args...; kwargs...)

@deprecate siteindex(args...; kwargs...) siteind(args...; kwargs...)

@deprecate linkindex(args...; kwargs...) linkind(args...; kwargs...)

