function NDTensors.svd_catch_error(A::CuMatrix; alg::String="JacobiAlgorithm")
  if alg == "JacobiAlgorithm"
    alg = CUDA.CUSOLVER.JacobiAlgorithm()
  else
    alg = CUDA.CUSOLVER.QRAlgorithm()
  end
  return NDTensors.svd_catch_error(A, alg)
end

function NDTensors.svd_catch_error(A::CuMatrix, ::CUDA.CUSOLVER.JacobiAlgorithm)
  USV = try
    svd(A; alg=CUDA.CUSOLVER.JacobiAlgorithm())
  catch
    return nothing
  end
  return USV
end

function NDTensors.svd_catch_error(A::CuMatrix, ::CUDA.CUSOLVER.QRAlgorithm)
  s = size(A)
  if s[1] < s[2]
    At = copy(Adjoint(A))

    USV = try
      svd(At; alg=CUDA.CUSOLVER.QRAlgorithm())
    catch
      return nothing
    end
    MV, MS, MU = USV
    USV = SVD(copy(MU), MS, Adjoint(MV))
  else
    USV = try
      svd(A; alg=CUDA.CUSOLVER.QRAlgorithm())
    catch
      return nothing
    end
  end
  return USV
end
