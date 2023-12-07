using Adapt: WrappedArray

const WrappedAbstractSparseArray{T,N,A} = WrappedArray{T,N,<:AbstractSparseArray{T,N}}

const SparseArrayLike{T,N} = Union{
  <:AbstractSparseArray{T,N},<:WrappedAbstractSparseArray{T,N}
}
