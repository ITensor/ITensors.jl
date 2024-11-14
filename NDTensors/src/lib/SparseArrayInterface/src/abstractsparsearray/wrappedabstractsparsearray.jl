using Adapt: WrappedArray

const WrappedAbstractSparseArray{T,N,A} = WrappedArray{
  T,N,<:AbstractSparseArray,<:AbstractSparseArray{T,N}
}

const AnyAbstractSparseArray{T,N} = Union{
  <:AbstractSparseArray{T,N},<:WrappedAbstractSparseArray{T,N}
}
