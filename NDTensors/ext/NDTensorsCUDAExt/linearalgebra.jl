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
