# ITensors.jl

ITensor is a library for rapidly creating correct and efficient
tensor network algorithms.

| **Documentation**|**Citation**|
|:----------------:|:----------:|
|[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://itensor.github.io/ITensors.jl/dev/)|[![SciPost](https://img.shields.io/badge/SciPost-10.21468-blue.svg)](https://scipost.org/SciPostPhysCodeb.4) [![arXiv](https://img.shields.io/badge/arXiv-2007.14822-b31b1b.svg)](https://arxiv.org/abs/2007.14822)|

|**Version**|**Download Statistics**|**Style Guide**|**License**|
|:---------:|:---------------------:|:-------------:|:---------:|
|[![version](https://juliahub.com/docs/ITensors/version.svg)](https://juliahub.com/ui/Packages/ITensors/P3pqL)|[![ITensor Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FITensors&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/ITensors)|[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)|[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ITensor/ITensors.jl/blob/main/LICENSE)|

The source code for ITensor can be found [on Github](https://github.com/ITensor/ITensors.jl).

Additional documentation can be found on the ITensor website [itensor.org](https://itensor.org/).

An ITensor is a tensor whose interface
is independent of its memory layout. ITensor indices are
objects which carry extra information and which
'recognize' each other (compare equal to each other).

The ITensor library also includes composable and extensible
algorithms for optimizing and transforming tensor networks, such as
matrix product state and matrix product operators, such as
the DMRG algorithm.

Development of ITensor is supported by the Flatiron Institute, a division of the Simons Foundation.

## News

- October 25, 2024: ITensors.jl v0.7 has been released. This is a major breaking change, since all of the MPS/MPO functionality from this package has been moved to [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl), along with all of the functionality of [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl). If you want to use MPS/MPO types and related functionality, such as `MPS`, `MPO`, `dmrg`, `siteinds`, `OpSum`, `op`, etc. you now must install and load the ITensorMPS.jl package. Additionally, if you are using ITensorTDVP.jl in your code, please change `using ITensorTDVP` to `using ITensorMPS`. ITensorMPS.jl has all of the same functionality as ITensorTDVP.jl, and ITensorTDVP.jl will be deprecated in favor of ITensorMPS.jl. **Note:** If you are using `ITensors.compile`, you must now install and load the ITensorMPS.jl package in order to trigger it to load properly, since it relies on running MPS/MPO functionality as example code for Julia to compile.

- May 9, 2024: A new package [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) has been released. We plan to move all of the MPS/MPO functionality in [ITensors.jl](https://github.com/ITensor/ITensors.jl) to [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl). For now, ITensorMPS.jl just re-exports the MPS/MPO functionality of ITensors.jl (as well as of [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl)), such as `dmrg`, `siteinds`, `MPS`, `MPO`, etc. To prepare for the change over to ITensorMPS.jl, please change `using ITensors` to `using ITensors, ITensorMPS` in any code that makes use of MPS/MPO functionality, and if you are using ITensorTDVP.jl change `using ITensorTDVP` to `using ITensorMPS` in your code.

- May 8, 2024: ITensors.jl v0.6 has been released. This version deletes the experimental "combine-contract" contraction backend, which was enabled by `ITensors.enable_combine_contract()`. This feature enabled performing ITensor contractions by first combining indices and then performing contractions as matrix multiplications, which potentially could lead to speedups for certain contractions involving higher-order QN-conserving tensors. However, the speedups weren't consistent with the current implementation, and this feature will be incorporated into the library in a more systematic way when we release our new non-abelian symmetric tensor backend.

- May 2, 2024: ITensors.jl v0.5 has been released. This version removes PackageCompiler.jl as a dependency and moves the package compilation functionality into a package extension. In order to use the `ITensors.compile()` function going forward, you need to install the PackageCompiler.jl package with `using Pkg: Pkg; Pkg.add("PackageCompiler")` and put `using PackageCompiler` together with `using ITensors` in your code.

- April 16, 2024: ITensors.jl v0.4 has been released. This version removes HDF5.jl as a dependency and moves the HDF5 read and write functions for ITensor, MPS, MPO, and other associated types into a package extension. To enable ITensor HDF5 features, install the HDF5.jl package with `using Pkg: Pkg; Pkg.add("HDF5")` and put `using HDF5` together with `using ITensors` in your code. Other recent changes include support for multiple GPU backends using package extensions.

- March 25, 2022: ITensors.jl v0.3 has been released. The main breaking change is that we no longer support versions of Julia below 1.6. Julia 1.6 is the long term support version of Julia (LTS), which means that going forward versions below Julia 1.6 won't be as well supported with bug fixes and improvements. Additionally, Julia 1.6 introduced many improvements including syntax improvements that we would like to start using with ITensors.jl, which becomes challenging if we try to support Julia versions below 1.6. See [here](https://www.oxinabox.net/2021/02/13/Julia-1.6-what-has-changed-since-1.0.html) and [here](https://julialang.org/blog/2021/03/julia-1.6-highlights/) for some nice summaries of the Julia 1.6 release.

-  Jun 09, 2021: ITensors.jl v0.2 has been released, with a few breaking changes as well as a variety of bug fixes
and new features. Take a look at the [upgrade guide](https://itensor.github.io/ITensors.jl/stable/UpgradeGuide_0.1_to_0.2.html)
for help upgrading your code.

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
Please note that right now, ITensors.jl requires that you use Julia v1.3 or later (since ITensors.jl relies on a feature that was introduced in Julia v1.3).

We recommend using ITensors.jl with Intel MKL in order to get the best possible performance. If you have not done so already, you can replace your current BLAS and LAPACK implementation with MKL by using the MKL.jl package. Please follow the instructions [here](https://github.com/JuliaComputing/MKL.jl).

## Documentation

- [**LATEST**](https://itensor.github.io/ITensors.jl/dev/) -- *documentation of the latest version.*

## Citation

If you use ITensor in your work, please cite the [ITensor Paper](https://www.scipost.org/SciPostPhysCodeb.4):

```bib
@article{ITensor,
	title={{The ITensor Software Library for Tensor Network Calculations}},
	author={Matthew Fishman and Steven R. White and E. Miles Stoudenmire},
	journal={SciPost Phys. Codebases},
	pages={4},
	year={2022},
	publisher={SciPost},
	doi={10.21468/SciPostPhysCodeb.4},
	url={https://scipost.org/10.21468/SciPostPhysCodeb.4},
}
```

and associated "Codebase Release" for the version you have used. The current one is

```bib
@article{ITensor-r0.3,
	title={{Codebase release 0.3 for ITensor}},
	author={Matthew Fishman and Steven R. White and E. Miles Stoudenmire},
	journal={SciPost Phys. Codebases},
	pages={4-r0.3},
	year={2022},
	publisher={SciPost},
	doi={10.21468/SciPostPhysCodeb.4-r0.3},
	url={https://scipost.org/10.21468/SciPostPhysCodeb.4-r0.3},
}
```

## ITensor Code Samples

### Basic Overview

ITensor construction, setting of elements, contraction, and addition.
Before constructing an ITensor, one constructs Index objects
representing tensor indices.

```julia
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

  # Contract over shared index j
  C = A * B

  @show hasinds(C,i,k,l) # = true

  D = random_itensor(k,j,i) # ITensor with random elements

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

```julia
using ITensors
let
  i = Index(10)           # index of dimension 10
  j = Index(20)           # index of dimension 20
  M = random_itensor(i,j)  # random matrix, indices i,j
  U,S,V = svd(M,i)        # compute SVD with i as row index
  @show M ≈ U*S*V         # = true

  nothing
end

# output

M ≈ U * S * V = true
```

### Singular Value Decomposition (SVD) of a Tensor

In this example, we create a random 4x4x4x4 tensor
and compute its SVD, temporarily treating the indices i and k
together as the "row" index and j and l as the "column" index
for the purposes of the SVD. The resulting factors can
be simply multiplied back together using the
ITensor `*` operation, which automatically recognizes
the matching indices between U and S, and between S and V
and contracts (sums over) them.

![](svd_tensor.png)

```julia
using ITensors
let
  i = Index(4,"i")
  j = Index(4,"j")
  k = Index(4,"k")
  l = Index(4,"l")
  T = random_itensor(i,j,k,l)
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

```julia
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

```julia
using ITensors, ITensorMPS
let
  # Create 100 spin-one indices
  N = 100
  sites = siteinds("S=1",N)

  # Input operator terms which define
  # a Hamiltonian matrix, and convert
  # these terms to an MPO tensor network
  # (here we make the 1D Heisenberg model)
  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 0.5,"S+",j,"S-",j+1
    os += 0.5,"S-",j,"S+",j+1
  end
  H = MPO(os,sites)

  # Create an initial random matrix product state
  psi0 = random_mps(sites)

  # Plan to do 5 passes or 'sweeps' of DMRG,
  # setting maximum MPS internal dimensions
  # for each sweep and maximum truncation cutoff
  # used when adapting internal dimensions:
  nsweeps = 5
  maxdim = [10,20,100,100,200]
  cutoff = 1E-10

  # Run the DMRG algorithm, returning energy
  # (dominant eigenvalue) and optimized MPS
  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
  println("Final energy = $energy")

  nothing
end

# output

After sweep 1 energy=-137.954199761732 maxlinkdim=9 maxerr=2.43E-16 time=9.356
After sweep 2 energy=-138.935058943878 maxlinkdim=20 maxerr=4.97E-06 time=0.671
After sweep 3 energy=-138.940080155429 maxlinkdim=92 maxerr=1.00E-10 time=4.522
After sweep 4 energy=-138.940086009318 maxlinkdim=100 maxerr=1.05E-10 time=11.644
After sweep 5 energy=-138.940086058840 maxlinkdim=96 maxerr=1.00E-10 time=12.771
Final energy = -138.94008605883985
```
You can find more examples of running `dmrg` and related algorithms [here](https://github.com/ITensor/ITensorMPS.jl/tree/main/examples).

