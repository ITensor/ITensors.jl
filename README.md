PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE FOR PREVIEW PURPOSES ONLY

THIS SOFTWARE IS SUBJECT TO BREAKING CHANGES AND NOT YET OFFICIALLY SUPPORTED

ITensors is a library for rapidly creating correct and efficient
tensor network algorithms. 

An ITensor is a tensor whose interface 
is independent of its memory layout. ITensor indices are
objects which carry extra information and which
'recognize' each other (compare equal to each other).

The ITensor library also includes composable and extensible 
algorithms for optimizing and transforming tensor networks, such as 
matrix product state and matrix product operators, such as
the DMRG algorithm.

Development of ITensor is supported by the Simons Foundation 
through the Flatiron Institute.

## Code Examples

### Singular Value Decomposition (SVD) of a Matrix

In this example, we create a random 10x20 matrix 
and compute its SVD. The resulting factors can 
be simply multiplied back together using the
ITensor `*` operation, which automatically recognizes
the matching indices between U and S, and between S and V
and contracts (sums over) them.

```Julia
using ITensors
let
  i = Index(10)           # index of dimension 10
  j = Index(20)           # index of dimension 20
  M = randomITensor(i,j)  # random matrix, indices i,j
  U,S,V = svd(M)          # compute SVD
  @show norm(M - U*S*V)   # ≈ 0.0
end
```

