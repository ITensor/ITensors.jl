using Metal
using NDTensors

using ITensors

# Here is an example of how to utilize NDTensors based tensors with CUDA datatypes
i = Index(20)
j = Index(5)
k = Index(78)
l = Index(62)

dim1 = (i, j, l)
dim2 = (j, k)

A = ITensor(NDTensors.generic_randn(MtlVector, dim(dim1)), dim1)
B = ITensor(NDTensors.generic_randn(MtlVector, dim(dim2)), dim2)
C = A * B

A = ITensor(Float64, dim1)
B = ITensor(Float64, dim2)

fill!(A, randn())
fill!(B, randn())

A = mtl(A)
B = mtl(B)

A * B

dim3 = (l,k);
dim4 = (i,)
C = ITensor(NDTensors.generic_randn(MtlVector{Float64}, dim(dim3)), dim3)
D = ITensor(NDTensors.generic_randn(MtlVector{Float64}, dim(dim4)), dim4)


f(A, B, C, D) = (A * B * C * D)[]
using Zygote

# grad = gradient(f, A, B, C, D)
# grad[1]
# grad[2]
# grad[3]

@show data(storage(B))
svd(B, (j))