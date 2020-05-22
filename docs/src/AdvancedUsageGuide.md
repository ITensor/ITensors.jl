
# Advanced ITensor usage guide

## Installing and updating ITensors.jl

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

We recommend using ITensors.jl with Intel MKL in order to get the 
best possible performance. If you have not done so already, you can 
replace the current BLAS and LAPACK implementation used by Julia with 
MKL by using the MKL.jl package. Please follow the instructions 
[here](https://github.com/JuliaComputing/MKL.jl).

To use the latest version of ITensors.jl, use `update ITensors`. 
We will commonly release new minor versions with bug fixes and 
improvements. However, make sure to double check before doing this, 
because new releases may be breaking.

To try the "development branch" of ITensors.jl (for example, if 
there is a feature or fix we added that hasn't been released yet), 
you can do `add ITensors#master`. This is generally not encouraged 
unless you know what you are doing.

## Writing code based on ITensors.jl

There are many ways you can write code based on ITensors.jl, ranging 
from using it in the REPL to writing a small script to making a 
package that depends on it.

## Developing ITensors.jl

To make your own changes to ITensors.jl, type `dev ITensors`. This 
will create a local clone of the Github repository in the directory 
`~/.julia/dev/ITensors`. Changes to that directory will be reflected 
when you do `using ITensors` in a new session.

We highly recommend using the Revise package when you are developing 
packages, which automatically detects changes you are making in a 
package so you can edit code and not have to restart your Julia 
session.

## Compiling ITensors.jl

You might notice that the time to load ITensors.jl (with `using 
ITensors`) and the time to run your first few ITensors commands is 
slow. This is due to Julia's just-in-time (JIT) compilation.

 - Precompilation
 - Staying in the same Julia session with Revise
 - Using PackageCompile

## Multithreading Support

There are two possible sources of parallelization available in ITensors.jl, both external to the package right now. These are:
 - BLAS/LAPACK multithreading (through whatever flavor you are using, i.e. OpenBLAS or MKL).
 - The Strided.jl package, which implements a multithreaded array permutation.

The BLAS/LAPACK multithreading can be controlled in the usual way with environment variables, or within Julia. So for example, to control from Julia, you would do:
```julia
julia> using LinearAlgebra

julia> BLAS.vendor()  # Check which BLAS you are using
:mkl

julia> BLAS.set_num_threads(4)

julia> ccall((:MKL_GET_MAX_THREADS, Base.libblas_name), Cint, ())
4

julia> BLAS.set_num_threads(2)

julia> ccall((:MKL_GET_MAX_THREADS, Base.libblas_name), Cint, ())
2
```
(if you are using OpenBLAS, I think you would get the number of threads with `ccall((:openblas_get_num_threads, Base.libblas_name), Cint, ())` but I haven't checked).

Alternatively, you can use environment variables, so at your command line prompt you would use:
```
~ export MKL_NUM_THREADS=4
```
if you are using MKL or
```
~ export OPENBLAS_NUM_THREADS=4
```
if you are using OpenBLAS. We would highly recommend using MKL (you can easily add MKL support to an existing Julia build with https://github.com/JuliaComputing/MKL.jl), especially if you are using an Intel chip. In general, we have not found MKL/OpenBLAS multithreading to help much in the context of DMRG, and I would be very surprised if it scaled to 20 cores. How well it scales would depend highly on the problem you are studying, and would require your calculation to be vastly dominated by matrix multiplications (which is not always the case, especially if you are using QN conservation).

Then, a separate level of mutlithreading could be turned on, which is native Julia multithreading. Right now in ITensors.jl, this would only control array permutation functions we use from [Strided.jl](https://github.com/Jutho/Strided.jl). You would set it with the environment variable `JULIA_NUM_THREADS`, for example:
```julia
julia> Threads.nthreads() # By default it is probably off
1
```
Then if you set `export JULIA_NUM_THREADS=4` at your command line, you would see the next time you start up Julia:
```julia
julia> Threads.nthreads()
4
```
Currently, we have not found that using that kind of multithreading has helped either, and I have seen it cause slowdowns, possibly because it is competing with BLAS multithreading (Jutho, the author of the package, is aware of those problems and it will hopefully improve in the future).

On top of that, we hope to incorporate our own multithreading with Julia's native multithreading capabilities, for example to parallelize over block sparse contractions. We have that implemented in the C++ version of ITensor, and it works very well (for certain problems, it did in fact scale up to 20 cores).

## ITensor type design and writing performant code

Advanced users might notice something strange about the definition
of the ITensor type, that it is often not "type stable". Some of 
this is by design. The definition for ITensor is:
```julia
mutable struct ITensor{N}
  ::IndexSet{N}
  ::TensorStorage
end
```
These are both abstract types, which is something that is generally 
discouraged for peformance.

This has a few disadvantages. Some code that you might expect to be 
type stable, like `getindex`, is not, for example:
```julia
@code_warntype A[i=>1, j=>2]
```
Julia can't know ahead of time, based on the inputs, what the type 
of the output is (though at runtime, the output has a concrete type).

So why is it designed this way? The main reason is to allow more 
generic code. This allows us to have code like:
```julia
A = randomITensor(i', i)
A .*= 2+1im
```
Here, the type of the storage of A is changed in-place. More 
generally, this allows ITensors to have more generic in-place 
functionality, so you can write code where you don't know what the s
torage is until runtime.

This can lead to certain types of code having perfomance problems, 
for example looping through ITensors can be slow:
```julia
A = randomITensor(i', i)
for n in 1:dim(A)
  A[n] = 2 * A[n]
end
```
However, this is fast:
```julia
A .*= 2
```
How does this work? It relies on a "function barrier" technique. 
Julia compiles functions "just-in-time", so that calls an inner 
function written in terms of a type-stable Tensor type from the 
package NDTensors. That is the function ultimately being called. 
The main overhead is that Julia has to determine which function 
to call at runtime.

You can look at a simple example to see how this works.
```julia
struct MyMatrix
  data
end

import Base: *

# This is our function barrier
A::MyMatrix * B::MyMatrix = A.data * B.data

for d in 5:5:40
  a = randn(d,d)
  b = randn(d,d)

  A = MyMatrix(a)
  B = MyMatrix(b)

  @btime $a*$b
  @btime $A*$B
end
```
However,
```julia
d = 100

a = randn(d,d)

A = MyMatrix(a)

function Base.sum(A::MyMatrix)
  t = 0.0
  for i in 1:length(A.data)
    t += A.data[i]
  end
  return A
end
```

One can make code type stable by putting explicit type declerations, 
if they know the type beforehand. For example:
```julia
function Base.sum(A::MyMatrix)
  t::Float64 = 0.0
  for i in 1:length(A.data)
    t += A.data[i]
  end
  return A
end
```

Therefore, users should keep this in mind when they are writing 
ITensors.jl code, and we warn that explicitly looping over large 
ITensor by individual elements should be avoiding in performance 
critical sections of your code. However, they shouldn't be too
worried about this, as rest assured high level ITensor functions
are still performant.

## ITensor in-place operations

In-place operations can help with optimizating code, when the
memory is preallocated.

The main way to access this in ITensor is through broadcasting.
For example:
```julia
A .*= 2
```
Internally, this is rewritten by Julia as a call to `broadcast!`.
ITensors.jl overloads this call (or more specifically, a lower
level function `copyto!`). Then, this call is rewritten as
```julia
map!(x -> 2*x, A, A)
```

Additionally, ITensors makes the unique choice that:
```julia
C .= A .* B
```
is interpreted as an in-place tensor contraction. What this means
is that this calls a function:
```julia
mul!(C, A, B)
```
(likely to be given an alternative name `contract!`) which contracts
`A` and `B` into the pre-allocated memory `C`.

Because of the design of the ITensor type (see the section above),
there is some flexibility we take in allocating memory for users.
For example, if the storage type is more narrow than the result,
for convenience we will expand it in-place.

## NDTensors and ITensors

ITensors.jl is built on top of another, more traditional tensor 
library called NDTensors. NDTensors implements AbstractArrays with 
a variety of sparse storage types, with more to come in the future.

NDTensors implements functionality like permutation of dimensions, 
fast get and set index, broadcasting, and tensor contraction (where 
labels of the dimensions must be specified).

For example:
```julia
using ITensors
using NDTensors

T = Tensor(2,2,2)
i = Index(2)
T = Tensor(i,i',i')  # The identifiers are ignored, just interpreted as above
```
To make performant ITensor code (refer to the the previous section 
on type stability and function barriers), ITensor storage data and 
indices are passed by reference into Tensors, where the performance 
critical operations are performed.

