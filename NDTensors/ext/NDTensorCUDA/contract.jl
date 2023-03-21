# CUDA generic matmul
function backend_cutensor()
    return gemm_backend[] = :CUTENSOR
  end
  function backend_cublas()
    return gemm_backend[] = :CUBLAS
  end
  
  @inline function NDTensors.auto_select_backend(
    ::Type{CuVecOrMat},
    ::Type{CuVecOrMat},
    ::Type{CuVecOrMat},
  )
    return GemmBackend(:CUBLAS)
  end

  # CUDA generic matmul
function _gemm!(
    ::GemmBackend{:GenericCUDA},
    tA,
    tB,
    alpha,
    A::AbstractVecOrMat,
    B::AbstractVecOrMat,
    beta,
    C::AbstractVecOrMat,
  )
    println("In gemm! in cuda contract.jl")
    return C
  end