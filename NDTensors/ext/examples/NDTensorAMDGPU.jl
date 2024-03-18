## Code adapted from NDTensors/ext/examples/NDTensorCUDA.jl

using AMDGPU
using NDTensors
using ITensors
using ITensors: Index, ITensor, orthogonalize, qr, siteinds, svd
using Test: @test

function main()
  # using ITensorGPU
  cpu = NDTensors.cpu
  gpu = NDTensors.roc
  # Here is an example of how to utilize NDTensors based tensors with AMDGPU datatypes
  i = Index(2)
  j = Index(5)
  k = Index(3)
  l = Index(6)

  dim1 = (i, j, l)
  dim2 = (j, k)

  # Create 2 ITensors with AMDGPU backends
  A = ITensor(randomTensor(ROCArray, dim1))
  B = ITensor(randomTensor(ROCArray, dim2))

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

  # Check that backend of contraction is GPU
  @test A * A ≈ cpu(cA * cA)
  @test B * B ≈ cpu(cB * cB)
  @test A * B ≈ cpu(cA * cB)
  @test B * A ≈ cpu(cB * cA)

  dim3 = (l, k)
  dim4 = (i,)
  cC = ITensor(randomTensor(ROCArray{Float64,AMDGPU.Runtime.Mem.HIPBuffer}, dim3))
  cD = ITensor(Tensor(ROCArray{Float32}, dim4))
  fill!(cD, randn())

  # Its possible to compute QR of GPU tensor
  cq = qr(cA, (i,), (j, l))
  A ≈ cpu(cq[1]) * cpu(cq[2])

  res = ITensors.svd(A, (i,), (j, l))
  @show res
end

## running the main function with Float64
main()
