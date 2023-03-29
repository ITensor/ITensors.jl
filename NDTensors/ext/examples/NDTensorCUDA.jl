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

A = ITensor(NDTensors.generic_randn(CuVector, dim(dim1)), dim1)
B = ITensor(NDTensors.generic_randn(CuVector, dim(dim2)), dim2)
C = A * B

A = ITensor(Float32, dim1)
B = ITensor(Float64, dim2)

fill!(A, randn())
fill!(B, randn())

A = NDTensors.cu(A)
B = NDTensors.cu(B)

A * B

dim3 = (l, k);
dim4 = (i,)
C = ITensor(NDTensors.generic_randn(CuVector, dim(dim3)), dim3)
D = ITensor(Tensor(CuVector, dim4))
fill!(D, randn())

f(A, B, C, D) = (A * B * C * D)[]
using Zygote

grad = gradient(f, A, B, C, D)
grad[1]
grad[2]
grad[3]

ITensors.qr(A, (i,), (j,l,))

typeof(storage(A))
## This doesn't yet work baceuse making things like onehot create vectors instead of 
## CuVectors...
ITensors.svd(A, (i,), (j,l))

s = ITensors.siteinds("S=1/2", 20)
m = randomMPS(s; linkdims=50)
@which NDTensors.cu(m)
cm = NDTensors.cu(m);

typeof(storage(m[1]))
typeof(storage(cm[1]))

inner(cm', cm)

H = NDTensors.cu(randomMPO(s))
inner(cm', H, cm)

#This currently doesn't work without cu because orthogonalize transforms ITensors back to CPU vectors
cm = NDTensors.cu(orthogonalize(cm, 1))
H = NDTensors.cu(orthogonalize(H, 1))

@show storage(cm[1])
@show storage(H[1])

inner(cm', H, cm)