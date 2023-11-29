using Metal: MtlVector, mtl, @allowscalar
using NDTensors

using ITensors: ITensor, Index, randomITensor
using Test: @test
using Zygote: gradient

function main()
  # Here is an example of how to utilize NDTensors based tensors with CUDA datatypes
  i = Index(20)
  j = Index(5)
  k = Index(78)
  l = Index(62)

  dim1 = (i, j, l)
  dim2 = (j, k)

  cA = ITensor(NDTensors.generic_randn(MtlVector{Float32}, dim(dim1)), dim1)
  cB = ITensor(NDTensors.generic_randn(MtlVector{Float32}, dim(dim2)), dim2)
  cC = cA * cB

  cpu = NDTensors.cpu
  A = cpu(cA)
  B = cpu(cB)

  @test A * B ≈ cpu(cC)

  dim3 = (l, k)
  dim4 = (i,)

  cC = mtl(randomITensor(Float32, dim3))
  cD = mtl(randomITensor(Float32, dim4))

  f(A, B, C, D) = (A * B * C * D)[]

  grad = gradient(f, cA, cB, cC, cD)
  @test grad[2] ≈ cA * cC * cD
end

main()
