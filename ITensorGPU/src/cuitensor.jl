import ITensors.NDTensors: NeverAlias, AliasStyle, AllowAlias
import ITensors: ITensor
import CUDA: CuArray

function cuITensor(eltype::Type{<:Number}, inds::IndexSet)
  return itensor(
    NDTensors.default_storagetype(CuVector{eltype,default_buffertype()})(dim(inds)), inds
  )
end
cuITensor(::Type{T}, inds::Index...) where {T<:Number} = cuITensor(T, IndexSet(inds...))

cuITensor(is::IndexSet) = cuITensor(Float64, is)
cuITensor(inds::Index...) = cuITensor(IndexSet(inds...))

cuITensor() = ITensor()
function cuITensor(x::S, inds::IndexSet{N}) where {S<:Number,N}
  d = NDTensors.Dense{S,CuVector{S,default_buffertype()}}(
    fill!(CuVector{S,default_buffertype()}(undef, dim(inds)), x)
  )
  return ITensor(d, inds)
end
cuITensor(x::S, inds::Index...) where {S<:Number} = cuITensor(x, IndexSet(inds...))

function ITensor(
  as::AliasStyle,
  eltype::Type{<:Number},
  A::CuArray{<:Number},
  inds::Indices{Index{Int}};
  kwargs...,
)
  length(A) ≠ dim(inds) && throw(
    DimensionMismatch(
      "In ITensor(::CuArray, inds), length of AbstractArray ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))",
    ),
  )
  data = CuArray{eltype}(as, A)
  return itensor(Dense(data), inds)
end

# Fix ambiguity error
function ITensor(
  as::NDTensors.AliasStyle, eltype::Type{<:Number}, A::CuArray{<:Number}, inds::Tuple{}
)
  length(A) ≠ dim(inds) && throw(
    DimensionMismatch(
      "In ITensor(::CuArray, inds), length of AbstractArray ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))",
    ),
  )
  data = CuArray{eltype}(as, A)
  return itensor(Dense(data), inds)
end

# Helper functions for different view behaviors
CuArray{ElT,N}(::NeverAlias, A::AbstractArray) where {ElT,N} = CuArray{ElT,N}(A)
function CuArray{ElT,N}(::AllowAlias, A::AbstractArray) where {ElT,N}
  return convert(CuArray{ElT,N}, A)
end
function CuArray{ElT}(as::AliasStyle, A::AbstractArray{ElTA,N}) where {ElT,N,ElTA}
  return CuArray{ElT,N}(as, A)
end

# TODO: Change to:
# (Array{ElT, N} where {ElT})([...]) = [...]
# once support for `VERSION < v"1.6"` is dropped.
# Previous to Julia v1.6 `where` syntax couldn't be used in a function name
function CuArray{<:Any,N}(as::AliasStyle, A::AbstractArray{ElTA,N}) where {N,ElTA}
  return CuArray{ElTA,N}(as, A)
end

cuITensor(data::Array, inds...) = cu(ITensor(data, inds...))

cuITensor(data::CuArray, inds...) = ITensor(data, inds...)

cuITensor(A::ITensor) = cu(A)

function randomCuITensor(::Type{S}, inds::Indices) where {S<:Real}
  T = cuITensor(S, inds)
  randn!(T)
  return T
end
function randomCuITensor(::Type{S}, inds::Indices) where {S<:Complex}
  Tr = cuITensor(real(S), inds)
  Ti = cuITensor(real(S), inds)
  randn!(Tr)
  randn!(Ti)
  return complex(Tr) + im * Ti
end
function randomCuITensor(::Type{S}, inds::Index...) where {S<:Number}
  return randomCuITensor(S, IndexSet(inds...))
end
randomCuITensor(inds::IndexSet) = randomCuITensor(Float64, inds)
randomCuITensor(inds::Index...) = randomCuITensor(Float64, IndexSet(inds...))

CuArray(T::ITensor) = CuArray(tensor(T))

function CuArray{ElT,N}(T::ITensor, is::Vararg{Index,N}) where {ElT,N}
  ndims(T) != N && throw(
    DimensionMismatch(
      "cannot convert an $(ndims(T)) dimensional ITensor to an $N-dimensional CuArray."
    ),
  )
  TT = tensor(permute(T, is...; allow_alias=true))
  return CuArray{ElT,N}(TT)::CuArray{ElT,N}
end

function CuArray{ElT}(T::ITensor, is::Vararg{Index,N}) where {ElT,N}
  return CuArray{ElT,N}(T, is...)
end

function CuArray(T::ITensor, is::Vararg{Index,N}) where {N}
  return CuArray{eltype(T),N}(T, is...)::CuArray{<:Number,N}
end

CUDA.CuMatrix(A::ITensor) = CuArray(A)

function CuVector(A::ITensor)
  if ndims(A) != 1
    throw(DimensionMismatch("Vector() expected a 1-index ITensor"))
  end
  return CuArray(A)
end

function CuMatrix(T::ITensor, i1::Index, i2::Index)
  ndims(T) != 2 &&
    throw(DimensionMismatch("ITensor must be order 2 to convert to a Matrix"))
  return CuArray(T, i1, i2)
end
