[![Build Status](https://travis-ci.org/ITensor/ITensors.jl.svg?branch=master)](https://travis-ci.org/ITensor/ITensors.jl) [![codecov](https://codecov.io/gh/ITensor/ITensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensors.jl)

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

### Basic Overview

Here is a basic intro overview of making 
ITensors, setting some elements, contracting, and adding
ITensors. See further examples below for detailed
detailed examples of these operations and more.

```Julia
using ITensors
let
  i = Index(3,"i")
  j = Index(5,"j")
  k = Index(4,"k")
  l = Index(7,"l")

  A = ITensor(i,j,k)
  B = ITensor(j,l)

  A[i(1),j(1),k(1)] = 11.1
  A[i(2),j(1),k(2)] = 21.2
  A[i(3),j(1),k(1)] = 31.1
  # ...

  # contract over index j
  C = A*B

  @show hasinds(C,i,k,l) # == true

  D = randomITensor(k,j,i)

  # add two ITensors
  R = A+D

end

```

### Making Tensor Indices

Before making an ITensor, you have to define its indices.
Tensor indices in ITensors.jl are themselves objects that 
carry extra information beyond just their dimension.

```Julia
using ITensors
let
  i = Index(3)     # Index of dimension 3
  @show dim(i)     # i = 3

  j = Index(5,"j") # Index with a tag "j"

  s = Index(2,"Site,n=1") # Index with two tags,
                          # "Site" and "n=1"
  @show hastags(s,"Site") # hasTags(s,"Site") = true
  @show hastags(s,"n=1")  # hasTags(s,"n=1") = true
end
```

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

### Singular Value Decomposition (SVD) of a Tensor

In this example, we create a random 4x4x4x4 tensor 
and compute its SVD, temporarily treating the first
and third indices as the "row" index and the second
and fourth indices as the "column" index for the purposes
of the SVD. The resulting factors can 
be simply multiplied back together using the
ITensor `*` operation, which automatically recognizes
the matching indices between U and S, and between S and V
and contracts (sums over) them.

```Julia
using ITensors
let
  i = Index(4,"i")
  j = Index(4,"j")
  k = Index(4,"k")
  l = Index(4,"l")
  T = randomITensor(i,j,k,l)
  U,S,V = svd(T,(i,k))
  @show inds(U)
  @show inds(V)
  @show norm(T - U*S*V)   # ≈ 0.0
end
```
