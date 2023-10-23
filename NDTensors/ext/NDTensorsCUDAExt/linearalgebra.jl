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

function NDTensors.truncate!(P::CuArray; kwargs...)
  cpuP = NDTensors.cpu(P)
  value = NDTensors.truncate!(cpuP; kwargs...)
  P = adapt(typeof(P), cpuP)
  return value;
end