# Multithreading

Most modern computers, including laptops, have multiple cores (processing units) which can be used
to perform multiple tasks at the same time and therefore speed up computations.
Multithreading is a form of shared memory parallelism that makes use of these multiple cores that
you may have available.

There are three primary sources of parallelization available to ITensors.jl. These are:
 - BLAS/LAPACK multithreading (through whatever flavor you are using, i.e. OpenBLAS or MKL).
 - The Strided.jl package, which implements efficient multithreaded dense array permutations.
 - Block sparse multithreading (currently only for block sparse contractions) implemented in the NDTensors.jl package.

First, you can obtain the number of threads that are available to you with:
```julia
julia> Sys.CPU_THREADS
6
```

If your computations are dominated by large dense tensors, you likely want to make use of BLAS multithreading
in order to multithread dense matrix multiplications and other linear algebra methods like SVD
and QR decompositions. This will be on by default. The BLAS/LAPACK multithreading can be controlled
in the usual way with environment variables such as by starting Julia with:
```
$ MKL_NUM_THREADS=4 julia # Set the number of MKL threads to 4

$ OPENBLAS_NUM_THREADS=4 julia # Set the number of OpenBLAS threads to 4

$ OMP_NUM_THREADS=4 julia # Set the number of OpenMP threads to 4, which will be used by MKL or OpenBLAS if they are not specifically set
```
or at runtime from within Julia:
```julia
julia> using LinearAlgebra

julia> BLAS.vendor()  # Check which BLAS you are using
:mkl

julia> using ITensors

julia> BLAS.get_num_threads()
6

julia> BLAS.set_num_threads(4)

julia> BLAS.get_num_threads()
4
```
Note that in Julia v1.6, you will be able to use the command `using LinearAlgebra; BLAS.get_num_threads()`.

We would highly recommend using MKL (see the installation instructions for how to do that), especially if you
are using an Intel chip. How well BLAS multithreading will work depends on how much your
calculations are dominated by large dense matrix operations (which is not always the case,
especially if you are using QN conservation).

Currently, ITensors.jl makes use of the package [Strided.jl](https://github.com/Jutho/Strided.jl)
for performant dense array permutations. It also provides multithreaded array permutations.
If you start Julia with multiple threads, Strided multithreading is on by default:
```julia
$ julia -t 4

julia> Threads.nthreads()
4

julia> ITensors.Strided.get_num_threads()
4
```
We find that this threading competes with BLAS threading as well as ITensors.jl's own block sparse
multithreading, so if you are using Julia with multiple threads you may want to disable
Strided.jl's threading with:
```julia
julia> ITensors.Strided.disable_threads()
1

julia> ITensors.Strided.get_num_threads()
1
```
in favor of either BLAS threading or ITensors.jl's block sparse threading.

Additionally, ITensors.jl, through the [NDTensors.jl](https://github.com/ITensor/NDTensors.jl) library,
provides multithreaded block sparse operations. By default, this kind of threading is disabled.
If your computations involve QN conserving tensors, you may want to consider enabling block sparse
multithreading as described below.

```@docs
ITensors.enable_threaded_blocksparse
```

Here is a simple example of using block sparse multithreading to speed up a sparse
tensor contraction:
```julia
using BenchmarkTools
using ITensors
using LinearAlgebra

function main(; d = 20, order = 4)
  BLAS.set_num_threads(1)
  ITensors.Strided.set_num_threads(1)

  println("#################################################")
  println("# order = ", order)
  println("# d = ", d)
  println("#################################################")
  println()

  i(n) = Index(QN(0) => d, QN(1) => d; tags = "i$n")
  is = IndexSet(i, order ÷ 2)
  A = randomITensor(is'..., dag(is)...)
  B = randomITensor(is'..., dag(is)...)

  ITensors.enable_threaded_blocksparse(false)

  println("Serial contract:")
  @disable_warn_order begin
    C_contract = @btime $A' * $B samples = 5
  end
  println()

  println("Threaded contract:")
  @disable_warn_order begin
    ITensors.enable_threaded_blocksparse(true)
    C_threaded_contract = @btime $A' * $B samples = 5
    ITensors.enable_threaded_blocksparse(false)
  end
  println()
  @show C_contract ≈ C_threaded_contract
  return nothing
end

main(d = 20, order = 4)
```
which outputs the following on a laptop with 6 threads, starting Julia with
5 threads:
```
julia> main(d = 20, order = 4)
#################################################
# order = 4
# d = 20
#################################################

Threads.nthreads() = 5
Sys.CPU_THREADS = 6
BLAS.get_num_threads() = 1
ITensors.Strided.get_num_threads() = 1

Serial contract:
  21.558 ms (131 allocations: 7.34 MiB)

Threaded contract:
  5.934 ms (446 allocations: 7.37 MiB)

C_contract ≈ C_threaded_contract = true
```

Here is a full example of making use of block sparse multithreading to speed up
a DMRG calculation:
```julia
using ITensors
using LinearAlgebra
using Random

include(joinpath(ITensors.examples_dir(), "src", "electronk.jl"))
include(joinpath(ITensors.examples_dir(), "src", "hubbard.jl"))

function main(; Nx::Int=6, Ny::Int=3, U::Float64=4.0, t::Float64=1.0,
                maxdim::Int=3000, conserve_ky=true,
                nsweeps=10, blas_num_threads=1, strided_num_threads=1,
                threaded_blocksparse=true, outputlevel=1,
                seed=1234)
  Random.seed!(seed)
  ITensors.Strided.set_num_threads(strided_num_threads)
  BLAS.set_num_threads(blas_num_threads)
  ITensors.enable_threaded_blocksparse(threaded_blocksparse)

  if outputlevel > 0
    @show Threads.nthreads()
    @show Sys.CPU_THREADS
    @show BLAS.get_num_threads()
    @show ITensors.Strided.get_num_threads()
    @show ITensors.using_threaded_blocksparse()
    println()
  end

  N = Nx * Ny

  maxdim = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  cutoff = 1E-6
  noise = [1E-6, 1E-7, 1E-8, 0.0]

  sites = siteinds("ElecK", N; conserve_qns=true,
                   conserve_ky, modulus_ky=Ny)

  hubbard_ops = hubbard(Nx, Ny, t, U, ky=true)
  H = MPO(hubbard_ops, sites)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  if outputlevel > 0
    @show nnz(H[end÷2])
    @show nnzblocks(H[end÷2])
  end

  # Create starting state with checkerboard
  # pattern
  state = map(CartesianIndices((Ny, Nx))) do I
    return iseven(I[1]) ⊻ iseven(I[2]) ? "↓" : "↑"
  end
  display(state)

  psi0 = randomMPS(sites, state; linkdims=10)

  energy, psi = @time dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, outputlevel)

  if outputlevel > 0
    @show Nx, Ny
    @show t, U
    @show flux(psi)
    @show maxlinkdim(psi)
    @show energy
  end
  return nothing
end

println("Without threaded block sparse:\n")
main(; nsweeps=10, threaded_blocksparse=false)
println()
println("With threaded block sparse:\n")
main(; nsweeps=10, threaded_blocksparse=true)
println()
```
which when run on a laptop with 6 cores gives (after running once
with 1 or 2 sweeps to trigger compilation):
```
Without threaded block sparse:

Threads.nthreads() = 5
Sys.CPU_THREADS = 6
BLAS.get_num_threads() = 1
ITensors.Strided.get_num_threads() = 1
ITensors.using_threaded_blocksparse() = false

splitblocks = true
nnz(H[end ÷ 2]) = 67
nnzblocks(H[end ÷ 2]) = 67
After sweep 1 energy=-5.861157015737 maxlinkdim=78 time=0.633
After sweep 2 energy=-12.397725587986 maxlinkdim=200 time=6.980
After sweep 3 energy=-13.472412477115 maxlinkdim=400 time=14.257
After sweep 4 energy=-13.627475223585 maxlinkdim=800 time=9.801
After sweep 5 energy=-13.693628527487 maxlinkdim=2000 time=15.343
After sweep 6 energy=-13.711928225391 maxlinkdim=3000 time=24.260
After sweep 7 energy=-13.715575192226 maxlinkdim=3000 time=25.752
After sweep 8 energy=-13.716394378223 maxlinkdim=3000 time=25.907
After sweep 9 energy=-13.716535094341 maxlinkdim=3000 time=24.748
After sweep 10 energy=-13.716556326664 maxlinkdim=3000 time=24.185
171.903248 seconds (575.56 M allocations: 207.370 GiB, 9.37% gc time)
(Nx, Ny) = (6, 3)
(t, U) = (1.0, 4.0)
flux(psi) = QN(("Ky",0,3),("Nf",18,-1),("Sz",0))
maxlinkdim(psi) = 3000
energy = -13.716556326663678

With threaded block sparse:

Threads.nthreads() = 5
Sys.CPU_THREADS = 6
BLAS.get_num_threads() = 1
ITensors.Strided.get_num_threads() = 1
ITensors.using_threaded_blocksparse() = true

splitblocks = true
nnz(H[end ÷ 2]) = 67
nnzblocks(H[end ÷ 2]) = 67
After sweep 1 energy=-5.861157015735 maxlinkdim=78 time=1.117
After sweep 2 energy=-12.397725588011 maxlinkdim=200 time=6.587
After sweep 3 energy=-13.472412477124 maxlinkdim=400 time=12.094
After sweep 4 energy=-13.627475223588 maxlinkdim=800 time=8.760
After sweep 5 energy=-13.693628527488 maxlinkdim=2000 time=12.447
After sweep 6 energy=-13.711928225390 maxlinkdim=3000 time=17.401
After sweep 7 energy=-13.715575192226 maxlinkdim=3000 time=18.863
After sweep 8 energy=-13.716394378223 maxlinkdim=3000 time=19.258
After sweep 9 energy=-13.716535094341 maxlinkdim=3000 time=19.627
After sweep 10 energy=-13.716556326664 maxlinkdim=3000 time=18.446
134.604491 seconds (1.69 G allocations: 300.213 GiB, 18.02% gc time)
(Nx, Ny) = (6, 3)
(t, U) = (1.0, 4.0)
flux(psi) = QN(("Ky",0,3),("Nf",18,-1),("Sz",0))
maxlinkdim(psi) = 3000
energy = -13.71655632666368
```
Speedups should be greater for problems with tensors that are more
sparse or have more blocks (for example larger system sizes).

In addition, we plan to add more threading to other parts of the
code beyond contraction (such as SVD) and improve composibility with
other forms of threading like BLAS and Strided, so stay tuned!

