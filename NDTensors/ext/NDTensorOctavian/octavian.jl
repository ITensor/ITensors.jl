function NDTensors.backend_octavian()
  return NDTensors.gemm_backend[] = :Octavian
end

function _gemm!(
  ::NDTensors.GemmBackend{:Octavian},
  tA,
  tB,
  alpha,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta,
  C::AbstractVecOrMat,
)
  return Octavian.matmul!(
    C, tA == 'T' ? transpose(A) : A, tB == 'T' ? transpose(B) : B, alpha, beta
  )
end
