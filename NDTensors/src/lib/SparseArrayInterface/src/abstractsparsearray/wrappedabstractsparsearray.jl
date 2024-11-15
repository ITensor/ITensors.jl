using Adapt: WrappedArray
using LinearAlgebra: Adjoint, Transpose

const WrappedAbstractSparseArray{T,N,A} = WrappedArray{
  T,N,<:AbstractSparseArray,<:AbstractSparseArray{T,N}
}

const AnyAbstractSparseArray{T,N} = Union{
  <:AbstractSparseArray{T,N},<:WrappedAbstractSparseArray{T,N}
}

function stored_indices(a::Adjoint)
  return Iterators.map(I -> CartesianIndex(reverse(Tuple(I))), stored_indices(parent(a)))
end
stored_length(a::Adjoint) = stored_length(parent(a))
sparse_storage(a::Adjoint) = Iterators.map(adjoint, sparse_storage(parent(a)))

function stored_indices(a::Transpose)
  return Iterators.map(I -> CartesianIndex(reverse(Tuple(I))), stored_indices(parent(a)))
end
stored_length(a::Transpose) = stored_length(parent(a))
sparse_storage(a::Transpose) = Iterators.map(transpose, sparse_storage(parent(a)))
