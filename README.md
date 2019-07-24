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

```Julia
using ITensors
let
  i = Index(10)
  j = Index(10)
  M = randomITensor(i,j)
  U,S,V = svd(M)
  @show norm(M - U*S*V)
end
```

