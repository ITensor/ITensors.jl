# Introduction

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/ITensors.jl/stable/) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/ITensors.jl/dev/) | [![Tests](https://github.com/ITensor/ITensors.jl/workflows/Tests/badge.svg)](https://github.com/ITensor/ITensors.jl/actions?query=workflow%3ATests) [![codecov](https://codecov.io/gh/ITensor/ITensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensors.jl) |

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

Development of ITensor is supported by the Flatiron Institute, a division of the Simons Foundation.


## Installation

The ITensors package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
~ julia
```

```julia
julia> ]

pkg> add ITensors
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("ITensors")
```
Please note that right now, ITensors.jl requires that you use Julia v1.4 or later (since we are using a feature that was introduced in Julia v1.4). We will work on supporting older minor versions.

We recommend using ITensors.jl with Intel MKL in order to get the best possible performance. If you have not done so already, you can replace your current BLAS and LAPACK implementation with MKL by using the MKL.jl package. Please follow the instructions [here](https://github.com/JuliaComputing/MKL.jl).

## Documentation

- [**STABLE**](https://itensor.github.io/ITensors.jl/stable/) --  **documentation of the most recently tagged version.**
- [**DEVEL**](https://itensor.github.io/ITensors.jl/dev/) -- *documentation of the in-development version.*

## Code Examples

### Basic Overview

ITensor construction, setting of elements, contraction, and addition.
Before constructing an ITensor, one constructs Index objects
representing tensor indices.

```jldoctest; output=false
using ITensors
let
  i = Index(3)
  j = Index(5)
  k = Index(2)
  l = Index(7)

  A = ITensor(i,j,k)
  B = ITensor(j,l)

  # Set elements of A
  A[i=>1,j=>1,k=>1] = 11.1
  A[i=>2,j=>1,k=>2] = -21.2
  A[k=>1,i=>3,j=>1] = 31.1  # can provide Index values in any order
  # ...

  # A[k(1),i(3),j(1)] = 31.1  # alternative notation

  # Contract over shared index j
  C = A * B

  @show hasinds(C,i,k,l) # = true

  D = randomITensor(k,j,i) # ITensor with random elements

  # Add two ITensors
  # must have same set of indices
  # but can be in any order
  R = A + D

  nothing
end

# output

hasinds(C, i, k, l) = true
```


### Singular Value Decomposition (SVD) of a Matrix

In this example, we create a random 10x20 matrix 
and compute its SVD. The resulting factors can 
be simply multiplied back together using the
ITensor `*` operation, which automatically recognizes
the matching indices between U and S, and between S and V
and contracts (sums over) them.

```jldoctest; output=false
using ITensors
let
  i = Index(10)           # index of dimension 10
  j = Index(20)           # index of dimension 20
  M = randomITensor(i,j)  # random matrix, indices i,j
  U,S,V = svd(M,i)        # compute SVD with i as row index
  @show M ≈ U*S*V         # = true

  nothing
end

# output

M ≈ U * S * V = true
```

### Singular Value Decomposition (SVD) of a Tensor

In this example, we create a random 4x4x4x4 tensor 
and compute its SVD, temporarily treating the first
and third indices (i and k) as the "row" index and the second
and fourth indices (j and l) as the "column" index for the purposes
of the SVD. The resulting factors can 
be simply multiplied back together using the
ITensor `*` operation, which automatically recognizes
the matching indices between U and S, and between S and V
and contracts (sums over) them.

```jldoctest; output=false
using ITensors
let
  i = Index(4,"i")
  j = Index(4,"j")
  k = Index(4,"k")
  l = Index(4,"l")
  T = randomITensor(i,j,k,l)
  U,S,V = svd(T,i,k)   # compute SVD with (i,k) as row indices (indices of U)
  @show hasinds(U,i,k) # = true
  @show hasinds(V,j,l) # = true
  @show T ≈ U*S*V      # = true

  nothing
end

# output

hasinds(U, i, k) = true
hasinds(V, j, l) = true
T ≈ U * S * V = true
```

### Tensor Indices: Tags and Prime Levels

Before making an ITensor, you have to define its indices.
Tensor Index objects carry extra information beyond just their dimension.

All Index objects carry a permanent, immutable id number which is 
determined when it is constructed, and allow it to be matched
(compare equal) with copies of itself.

Additionally, an Index can have up to four tag strings, and an
integer primelevel. If two Index objects have different tags or 
different prime levels, they do not compare equal even if they
have the same id.

Tags are also useful for identifying Index objects when printing
tensors, and for performing certain Index manipulations (e.g.
priming indices having certain sets of tags).

```jldoctest; output=false, filter=r"0x[0-9a-f]{16}"
using ITensors
let
  i = Index(3)     # Index of dimension 3
  @show dim(i)     # = 3
  @show id(i)      # = 0x5d28aa559dd13001 or similar

  ci = copy(i)
  @show ci == i    # = true

  j = Index(5,"j") # Index with a tag "j"

  @show j == i     # = false

  s = Index(2,"n=1,Site") # Index with two tags,
                          # "Site" and "n=1"
  @show hastags(s,"Site") # = true
  @show hastags(s,"n=1")  # = true

  i1 = prime(i) # i1 has a "prime level" of 1
                # but otherwise same properties as i
  @show i1 == i # = false, prime levels do not match

  nothing
end

# output

dim(i) = 3
id(i) = 0x5d28aa559dd13001
ci == i = true
j == i = false
hastags(s, "Site") = true
hastags(s, "n=1") = true
i1 == i = false
```

### DMRG Calculation

DMRG is an iterative algorithm for finding the dominant
eigenvector of an exponentially large, Hermitian matrix.
It originates in physics with the purpose of finding
eigenvectors of Hamiltonian (energy) matrices which model
the behavior of quantum systems.

```jldoctest; output=false, filter=[r"After sweep [1-5] energy=\-[0-9]{3}\.[0-9]{10,16} maxlinkdim=[0-9]{1,3} time=[0-9]{1,2}\.[0-9]{3}", r"Final energy = \-138\.[0-9]{10,16}"]
using ITensors
let
  # Create 100 spin-one indices
  N = 100
  sites = siteinds("S=1",N)

  # Input operator terms which define 
  # a Hamiltonian matrix, and convert
  # these terms to an MPO tensor network
  # (here we make the 1D Heisenberg model)
  ampo = AutoMPO()
  for j=1:N-1
    ampo += "Sz",j,"Sz",j+1
    ampo += 0.5,"S+",j,"S-",j+1
    ampo += 0.5,"S-",j,"S+",j+1
  end
  H = MPO(ampo,sites)

  # Create an initial random matrix product state
  psi0 = randomMPS(sites)

  # Plan to do 5 passes or 'sweeps' of DMRG,
  # setting maximum MPS internal dimensions 
  # for each sweep and maximum truncation cutoff
  # used when adapting internal dimensions:
  sweeps = Sweeps(5)
  maxdim!(sweeps, 10,20,100,100,200)
  cutoff!(sweeps, 1E-10)
  @show sweeps

  # Run the DMRG algorithm, returning energy 
  # (dominant eigenvalue) and optimized MPS
  energy, psi = dmrg(H,psi0, sweeps)
  println("Final energy = $energy")

  nothing
end

# output

sweeps = Sweeps
1 cutoff=1.0E-10, maxdim=10, mindim=1, noise=0.0E+00
2 cutoff=1.0E-10, maxdim=20, mindim=1, noise=0.0E+00
3 cutoff=1.0E-10, maxdim=100, mindim=1, noise=0.0E+00
4 cutoff=1.0E-10, maxdim=100, mindim=1, noise=0.0E+00
5 cutoff=1.0E-10, maxdim=200, mindim=1, noise=0.0E+00

After sweep 1 energy=-137.845841178879 maxlinkdim=9 time=8.538
After sweep 2 energy=-138.935378608196 maxlinkdim=20 time=0.316
After sweep 3 energy=-138.940079710492 maxlinkdim=88 time=1.904
After sweep 4 energy=-138.940086018149 maxlinkdim=100 time=4.179
After sweep 5 energy=-138.940086075413 maxlinkdim=96 time=4.184
Final energy = -138.94008607296038
```
