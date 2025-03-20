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

julia> using Strided

julia> Strided.get_num_threads()
4
```
We find that this threading competes with BLAS threading as well as ITensors.jl's own block sparse
multithreading, so if you are using Julia with multiple threads you may want to disable
Strided.jl's threading with:
```julia
julia> Strided.disable_threads()
1

julia> Strided.get_num_threads()
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
using Strided

function main(; d = 20, order = 4)
  BLAS.set_num_threads(1)
  Strided.set_num_threads(1)

  println("#################################################")
  println("# order = ", order)
  println("# d = ", d)
  println("#################################################")
  println()

  i(n) = Index(QN(0) => d, QN(1) => d; tags = "i$n")
  is = ntuple(i, order ÷ 2)
  A = random_itensor(is'..., dag(is)...)
  B = random_itensor(is'..., dag(is)...)

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
Strided.get_num_threads() = 1

Serial contract:
  21.558 ms (131 allocations: 7.34 MiB)

Threaded contract:
  5.934 ms (446 allocations: 7.37 MiB)

C_contract ≈ C_threaded_contract = true
```

In addition, we plan to add more threading to other parts of the
code beyond contraction (such as SVD) and improve composibility with
other forms of threading like BLAS and Strided, so stay tuned!

