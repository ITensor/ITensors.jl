function LinearAlgebra.mul!(
  C::MatrixStorageTensor, A::MatrixStorageTensor, B::MatrixStorageTensor
)
  mul!(storage(C), storage(A), storage(B))
  return C
end
