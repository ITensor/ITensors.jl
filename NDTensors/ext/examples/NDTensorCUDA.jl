using NDTensors
using CUDA: CUDA, CuVector, cu, reshape
using ITensors:
  Index, ITensor, randomMPO, randomMPS, inner, orthogonalize, qr, siteinds, svd
using Test: @test
using Zygote: gradient

function main()
  # using ITensorGPU
  cpu = NDTensors.cpu
  gpu = NDTensors.cu
  # Here is an example of how to utilize NDTensors based tensors with CUDA datatypes
  i = Index(2)
  j = Index(5)
  k = Index(3)
  l = Index(6)

  dim1 = (i, j, l)
  dim2 = (j, k)

  # Create  2 ITensors with CUDA backends (These will be made simpiler by randomITensor(CuVector) soon)
  A = ITensor(randomTensor(CuVector, dim1))
  B = ITensor(randomTensor(CuVector, dim2))
  # Contract the two tensors
  C = A * B
  A = cpu(A)
  B = cpu(B)
  @test cpu(C) ≈ A * B
  @test eltype(C) == Float64

  # Create 2 ITensors on CPU with different eltypes
  A = ITensor(Float32, dim1)
  B = ITensor(Float64, dim2)

  fill!(A, randn())
  fill!(B, randn())

  # Convert the ITensors to GPU
  cA = gpu(A)
  cB = gpu(B)

  #Check that backend of contraction is GPU
  @test A * A ≈ cpu(cA * cA)
  @test B * B ≈ cpu(cB * cB)
  @test A * B ≈ cpu(cA * cB)
  @test B * A ≈ cpu(cB * cA)

  dim3 = (l, k)
  dim4 = (i,)
  cC = ITensor(randomTensor(CuVector{Float64,CUDA.Mem.DeviceBuffer}, dim3))
  cD = ITensor(Tensor(CuVector{Float32}, dim4))
  fill!(cD, randn())

  # Create a function of 4 tensors on GPU
  f(A, B, C, D) = (A * B * C * D)[]

  #Use Zygote to take the gradient of the four tensors on GPU
  #Currently this code fails with CUDA.allowscalar(false)
  # Because of outer calling the _gemm! function which calls a 
  # generic implementation
  grad = gradient(f, cA, cB, cC, cD)
  @test cpu(cB * cC * cD) ≈ cpu(grad[1])
  @test (cB * cC * cD) ≈ grad[1]
  # Create a tuple of indices
  dims = size(grad[1])
  decomp = (dims[1], dims[2] * dims[3])
  # Reshape the CuVector of data into a matrix
  cuTensor_data = reshape(array(grad[1]), decomp)
  # Use cuBLAS to compute SVD of data
  U, S, V = svd(cuTensor_data)
  decomp = size(array(grad[2]))
  cuTensor_data = reshape(array(grad[2]), decomp)
  U, S, V = svd(cuTensor_data)

  # These things can take up lots of memory, look at memory usage here
  cuTensor_data = U = S = V = nothing
  GC.gc()
  CUDA.memory_status()

  # Get rid of the gradients and clean the CUDA memory
  CUDA.reclaim()
  CUDA.memory_status()

  # Its possible to compute QR of GPU tensor
  cq = qr(cA, (i,), (j, l))
  A ≈ cpu(cq[1]) * cpu(cq[2])

  ## SVD does not yet work with CUDA backend, see above on
  ## Converting ITensors to vectors and calling CUDA svd function
  ## CuVectors...
  #ITensors.svd(A, (i,), (j, l))

  s = siteinds("S=1/2", 8)
  m = randomMPS(s; linkdims=4)
  cm = gpu(m)

  @test inner(cm', cm) ≈ inner(m', m)

  H = randomMPO(s)
  cH = gpu(H)
  @test inner(cm', cH, cm) ≈ inner(m', H, m)

  m = orthogonalize(m, 1)
  cm = gpu(orthogonalize(cm, 1))
  @test inner(m', m) ≈ inner(cm', cm)

  H = orthogonalize(H, 1)
  cH = gpu(cH)

  @test inner(cm', cH, cm) ≈ inner(m', H, m)
end

## running the main function with Float64
main()
