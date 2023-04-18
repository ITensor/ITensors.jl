using CUDA
using NDTensors

using ITensors
using Test
# using ITensorGPU
# Here is an example of how to utilize NDTensors based tensors with CUDA datatypes
i = Index(2)
j = Index(5)
k = Index(3)
l = Index(6)

dim1 = (i, j, l)
dim2 = (j, k)

# Create  2 ITensors with CUDA backends (These will be made simpiler by randomITensor(CuVector) soon)
A = ITensor(NDTensors.generic_randn(CuVector, dim(dim1)), dim1)
B = ITensor(NDTensors.generic_randn(CuVector, dim(dim2)), dim2)
# Contract the two tensors
cpu = NDTensors.cpu
C = A * B
A = cpu(A)
B = cpu(B)
@test cpu(C) == A * B
@test eltype(C) == Float64

# Create 2 ITensors on CPU
A = ITensor(Float32, dim1)
B = ITensor(Float64, dim2)

fill!(A, randn())
fill!(B, randn())

# Convert the ITEnsors to GPU
cA = NDTensors.cu(A)
cB = NDTensors.cu(B)

#Check that backend of contraction is GPU
typeof(storage(A * B))
typeof(storage(cA * cB))
@test A * B == cpu(cA * cB)

dim3 = (l, k);
dim4 = (i,)
C = ITensor(NDTensors.generic_randn(CuVector, dim(dim3)), dim3)
D = ITensor(Tensor(CuVector, dim4))
fill!(D, randn())

# Create a function of 4 tensors on GPU
f(cA, cB, C, D) = (cA * cB * C * D)[]
using Pkg;
Pkg.add("Zygote");
using Zygote

#Use Zygote to take the gradient of the four tensors on GPU
grad = gradient(f, cA, cB, C, D)
#@show grad[1]
typeof(storage(grad[1]))
typeof(storage(grad[2]))
@test (cB * C * D) ≈ grad[1]

# Create a tuple of indices
decomp = (
  dim(NDTensors.ind(grad[1], 1)),
  dim(NDTensors.ind(grad[1], 2)) * dim(NDTensors.ind(grad[1], 3)),
)
# Reshape the CuVector of data into a matrix
cuTensor_data = CUDA.reshape(NDTensors.data(storage(grad[1])), decomp)
# Use cuBLAS to compute SVD of data
U, S, V = svd(cuTensor_data)
decomp = (dim(NDTensors.ind(grad[2], 1)), dim(NDTensors.ind(grad[2], 2)))
cuTensor_data = CUDA.reshape(NDTensors.data(storage(grad[2])), decomp)
U, S, V = svd(cuTensor_data)

# These things can take up lots of memory, look at memory usage here
cuTensor_data = nothing
CUDA.memory_status()

# Get rid of the gradients and clean the CUDA memory
CUDA.reclaim()
CUDA.memory_status()

# Its possible to compute QR of GPU tensor
cq = ITensors.qr(cA, (i,), (j, l))
q = ITensors.qr(A, (i,), (j, l))
#@show cq[1]
#@show cq[2]
A ≈ cpu(cq[1]) * cpu(cq[2])

## This doesn't yet work baceuse making things like onehot create vectors instead of 
## CuVectors...
#ITensors.svd(A, (i,), (j, l))

s = ITensors.siteinds("S=1/2", 8)
m = randomMPS(s; linkdims=4)
#@which NDTensors.cu(m)
cm = NDTensors.cu(m);

typeof(storage(m[1]))
typeof(storage(cm[1]))

inner(cm', cm)

H = NDTensors.cu(randomMPO(s))
inner(cm', H, cm)

cm = NDTensors.cu(orthogonalize(cm, 1))
H = NDTensors.cu(orthogonalize(H, 1))

#@show storage(cm[1])
#@show storage(H[1])

inner(cm', H, cm)

### TO run the NDTensorCUDA tests in the NDTensors test suite. use the following commands in the NDTensors directory.
if false # false so we don't have an infinite loop
  using Pkg
  Pkg.add("CUDA")
  using CUDA
  Pkg.test("NDTensors"; test_args=["cuda"])
end
## TODO create option to turn cuda tests on to allow the use of NDTensor.cu
