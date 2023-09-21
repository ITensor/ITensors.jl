function NDTensors.svd_catch_error(A::CuMatrix; alg="JacobiAlgorithm")
  if alg == "JacobiAlgorithm"
    alg = CUDA.CUSOLVER.JacobiAlgorithm()
  else
    alg = CUDA.CUSOLVER.QRAlgorithm()
  end
  USV = try
    svd(A; alg=alg)
  catch
    return nothing
  end
  return USV
end

using LinearAlgebra
function NDTensors.mul!!(CM::CuArray, AM::AbstractArray, BM::AbstractArray, α, β)
  AM = AM isa CuArray ? AM : copy(AM)
  BM = BM isa CuArray ? BM : copy(BM)
  @show typeof(AM)
  return mul!(CM, AM, BM, α, β)
end
