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

## TODO Here truncate does logical operations of the values in P
## So its more efficient to just make it a CPU vector and 
## convert back to GPU
function NDTensors.truncate!(P::CuArray; kwargs...)
  cpuP = NDTensors.cpu(P)
  value = NDTensors.truncate!(cpuP; kwargs...)
  P = adapt(typeof(P), cpuP)
  return value
end
