using CUDA
using NDTensors

using ITensors

# Here is an example of how to utilize NDTensors based tensors with CUDA datatypes
i = Index(20)
j = Index(5)
k = Index(78)
l = Index(62)

dim1 = (i, j, l)
dim2 = (j, k)

NDTensors.generic_randn(CuVector, 20)
@which ITensor(NDTensors.generic_randn(CuVector, dim(dim1)), dim1)
Array{Float64}(NDTensors.NeverAlias(), NDTensors.generic_randn(CuVector, 20))
A = ITensor(NDTensors.generic_randn(CuVector, dim(dim1)), dim1)
B = ITensor(NDTensors.generic_randn(CuVector, dim(dim2)), dim2)
C = A * B

A = ITensor(Float32, dim1)
B = ITensor(Float64, dim2)

fill!(A, randn())
fill!(B, randn())

A = cu(A)
B = cu(B)

A * B

dim3 = (l,k);
dim4 = (i,)
C = ITensor(NDTensors.generic_randn(CuVector, dim(dim3)), dim3)
D = ITensor(Tensor(CuVector, dim4))
fill!(D, randn())
E = A * B * C
storage(E)
storage(D)
D * E

f(A, B, C) = (A * B * C)[]
using Zygote
gradient(f, A, B, C)

@show data(storage(B))
svd(B, (j))