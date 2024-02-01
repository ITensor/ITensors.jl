# AMDGPU.jl does not provide these yet
struct JacobiAlgorithm end
struct QRAlgorithm end
struct ROCtoCPUSVD end

JACOBI_MAX_ITERATIONS = 500

function NDTensors.svd_catch_error(A::ROCMatrix; alg::String="jacobi_algorithm")
  if alg == "jacobi_algorithm"
    alg = JacobiAlgorithm()
  elseif alg == "qr_algorithm"
    alg = QRAlgorithm()
    error("QRAlgorithm on AMD coming soon")
  elseif alg == "roc_to_cpu_svd"
    alg = ROCtoCPUSVD()
  else
    error(
      "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
    )
  end

  return NDTensors.svd_catch_error(A, alg)
end

function NDTensors.svd_catch_error(A::ROCMatrix, ::JacobiAlgorithm)

  # TODO: gesvdj! may fail if there is no free device memory, so this tries to free
  # some if there is less than 1GB remaining. Not sure if this should be handled in
  # AMDGPU.jl or if we should call `AMDGPU.Runtime.Mem.alloc_or_retry!`
  if AMDGPU.Mem.free() < Int(1e9)
    dev = AMDGPU.device()
    pool = AMDGPU.HIP.memory_pool(dev)

    println("GPU free memory less than 1GB, trying to free before running SVD")
    GC.gc()
    AMDGPU.HIP.hipMemPoolTrimTo(pool, 0)
    println("after GC and pool trim: main free memory = ", AMDGPU.Mem.free() / 1e9, "GB, pool used mem = ", AMDGPU.HIP.used_memory(pool) / 1e9, "GB")
  end

  float_type = real(eltype(A))
  U, S, V, residual, n_sweeps, info = try
      # second parameter is the absolute tolerance, setting to <= 0 means
      # setting it to machine precision
      AMDGPU.rocSOLVER.gesvdj!(A, float_type(-1.0), Int32(JACOBI_MAX_ITERATIONS))
  catch
    println("caught error running rocSOLVER Jacobi Algorithm (gesvdj!)")
    return nothing
  end

  if info != 0
    return nothing
  end

  return (U, S, V)
end

function NDTensors.svd_catch_error(A::ROCMatrix, ::ROCtoCPUSVD)
  U, S, V = svd(NDTensors.cpu(A))
  return NDTensors.roc.((U, S, V))
end
