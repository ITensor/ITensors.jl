using Metal: MtlVector
using NDTensors
using NDTensors.MetalExtensions: mtl

using ITensors: ITensor, Index, randomITensor
using Test: @test
using Zygote: gradient

function main()
  cpu = NDTensors.cpu
  gpu = mtl
  # Here is an example of how to utilize NDTensors based tensors with CUDA datatypes
  i = Index(20)
  j = Index(5)
  k = Index(78)
  l = Index(62)

  dim1 = (i, j, l)
  dim2 = (j, k)

  ## MtlArrays only support Float32 arithmatic
  cA = ITensor(randomTensor(MtlVector{Float32}, dim1))
  cB = ITensor(randomTensor(MtlVector{Float32}, dim2))
  cC = cA * cB

  A = cpu(cA)
  B = cpu(cB)

  @test A * B ≈ cpu(cC)

  dim3 = (l, k)
  dim4 = (i,)

  cC = gpu(randomITensor(Float32, dim3))
  cD = gpu(randomITensor(Float32, dim4))

  f(A, B, C, D) = (A * B * C * D)[]

  grad = gradient(f, cA, cB, cC, cD)
  @test grad[2] ≈ cA * cC * cD
end

main()
