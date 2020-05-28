
# Advanced ITensor usage guide

## Installing and updating ITensors.jl

The ITensors package can be installed with the Julia package manager.
Assuming you have already downloaded Julia, which you can get
[here](https://julialang.org/downloads/), from the Julia REPL, 
type `]` to enter the Pkg REPL mode and run:
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
you can do `add ITensors#master`. You can switch back to the latest
released version with `add ITensors`. Using the development/master
branch is generally not encouraged unless you know what you are doing.

## Using ITensors.jl in the REPL

There are many ways you can write code based on ITensors.jl, ranging 
from using it in the REPL to writing a small script to making a 
package that depends on it.

For example, you can just start the REPL from your command line like:
```
~ julia
```
assuming you have an available version of Julia with the ITensors.jl
package installed. Then just type:
```julia
julia> using ITensors
```
and start typing ITensor commands. For example:
```julia
julia> i = Index(2, "i")
(dim=2|id=355|"i")

julia> A = randomITensor(i, i')
ITensor ord=2 (dim=2|id=355|"i") (dim=2|id=355|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=355|"i")
Dim 2: (dim=2|id=355|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 1.2320011464276275  1.8504245734277216
 1.0763652402177477  0.030353720156277037

julia> (A*dag(A))[]
3.9627443142240617
```

Note that there are some "gotchas" with working in the REPL like this.
Technically, all commands in the REPL are in the "global scope".
The global scope might not work as you would expect, for example:
```julia
julia> for _ in 1:3
         A *= 2
       end
ERROR: UndefVarError: A not defined
Stacktrace:
 [1] top-level scope at ./REPL[12]:2
 [2] eval(::Module, ::Any) at ./boot.jl:331
 [3] eval_user_input(::Any, ::REPL.REPLBackend) at /home/mfishman/software/julia-1.4.0/share/julia/stdlib/v1.4/REPL/src/REPL.jl:86
 [4] run_backend(::REPL.REPLBackend) at /home/mfishman/.julia/packages/Revise/AMRie/src/Revise.jl:1023
 [5] top-level scope at none:0
```
since the `A` inside the for-loop introduces a new local variable.
Some alternatives are to wrap that part of the code in a let-block
or a function:
```julia
julia> function f(A)
         for _ in 1:3
           A *= 2
         end
         A
       end
f (generic function with 1 method)

julia> A = f(A)
ITensor ord=2 (dim=2|id=355|"i") (dim=2|id=355|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=355|"i")
Dim 2: (dim=2|id=355|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 9.85600917142102   14.803396587421773
 8.610921921741982   0.2428297612502163
```
In this particular case, you can alternatively modify the ITensor
in-place:
```julia
julia> for _ in 1:3
         A ./= 2
       end

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=355|"i")
Dim 2: (dim=2|id=355|"i")'
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 1.2320011464276275  1.8504245734277216
 1.0763652402177477  0.030353720156277037
```

A common place you might accidentally come across this is the 
following:
```julia
julia> N = 4;

julia> sites = siteinds("S=1/2",N);

julia> ampo = AutoMPO();

julia> for j=1:N-1
         ampo += ("Sz", j, "Sz", j+1)
       end
ERROR: UndefVarError: ampo not defined
Stacktrace:
 [1] top-level scope at ./REPL[16]:2
 [2] eval(::Module, ::Any) at ./boot.jl:331
 [3] eval_user_input(::Any, ::REPL.REPLBackend) at /home/mfishman/software/julia-1.4.0/share/julia/stdlib/v1.4/REPL/src/REPL.jl:86
 [4] run_backend(::REPL.REPLBackend) at /home/mfishman/.julia/packages/Revise/AMRie/src/Revise.jl:1023
 [5] top-level scope at none:0
```
In this case, you can use `ampo .+= ("Sz", j, "Sz", j+1)`,
`add!(ampo, "Sz", j, "Sz", j+1)`, or wrap your code in a let-block
or function.

Take a look at Julia's documentation [here](https://docs.julialang.org/en/v1/manual/variables-and-scoping/)
for rules on scoping. Also note that this behavior is particular
to Julia v1.4 and below, and is expected to change in v1.5.

Note that the REPL is very useful for prototyping code quickly,
but working directly in the REPL and outside of functions can
cause sub-optimal performance. See Julia's [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/index.html)
for more information.

We recommend the package [OhMyREPL](https://kristofferc.github.io/OhMyREPL.jl/latest/) which adds syntax highlighting to the Julia REPL.

Once you start to have longer code, you will want to put your
code into a file. For example, you may have a short script
with one or more functions based on ITensors.jl:
```
# my_itensor_script.jl
using ITensors

function norm2(A::ITensor)
  return (A*dag(A))[]
end
```
Then, in the same directory as your script `my_itensor_script.jl`,
just type:
```julia
julia> include("my_itensor_script.jl");

julia> i = Index(2; tags="i");

julia> A = randomITensor(i', i);

julia> norm2(A)
[...]
```

As your code gets more complicated and has more files, it is good
to organize it into a project. That will be covered in the
next section.

## Make a project based on ITensors.jl

In this section, we will describe how to make a project based on
ITensors.jl.

!!! info "Coming soon"

    A guide to making your own project based on ITensors.jl is coming soon.

## Developing ITensors.jl

To make your own changes to ITensors.jl, type `dev ITensors`
in Pkg mode (by typing `]` at the Julia prompt). This 
will create a local clone of the Github repository in the directory 
`~/.julia/dev/ITensors`. Changes to that directory will be reflected 
when you do `using ITensors` in a new session.

We highly recommend using the [Revise](https://timholy.github.io/Revise.jl/stable/) package when you are developing 
packages, which automatically detects changes you are making in a 
package so you can edit code and not have to restart your Julia 
session.

!!! info "Coming soon"

    A more extended guide for contributing to ITensors.jl, including 
    contributing to the related NDTensors.jl as well as a style 
    guide, is coming soon.

## Compiling ITensors.jl

You might notice that the time to load ITensors.jl (with `using 
ITensors`) and the time to run your first few ITensor commands is 
slow. This is due to Julia's just-in-time (JIT) compilation.
Julia is compiling special versions of each function that is
being called based on the inputs that it gets at runtime. This
allows it to have fast code, often nearly as fast as fully compiled
languages like C++, while still being a dynamic language.

However, the long startup time can still be annoying. In this section,
we will discuss some strategies that can be used to minimize this
annoyance, for example:
 - Precompilation.
 - Staying in the same Julia session with Revise.
 - Using PackageCompile to compile ITensors.jl ahead of time.

Precompilation is performed automatically when you first install
ITensors.jl or update a version and run the command `using ITensors`
for the first time. For example, when you first use ITensors after
installation or updating, you will see:
```julia
julia> using ITensors
[ Info: Precompiling ITensors [9136182c-28ba-11e9-034c-db9fb085ebd5]
```
The process is done automatically, and
puts some compiled binaries in your `~/.julia` directory. The
goal is to decrease the time it takes when you first type
`using ITensors` in your next Julia session, and also the time
it takes for you to first run ITensor functions in a new
Julia session. This helps the startup time, but currently doesn't
help enough.
This is something both ITensors.jl and the Julia language will try
to improve over time.

To avoid this time, it is recommended that you work as much as you
can in a single Julia session. You should not need to restart your
Julia session very often. For example, if you are writing code in
a script, just `include` the file again which will pull in the new
changes to the script (the exception is if you change the definition
of a type you made, which would requiring restarting the REPL).

If you are working on a project, we highly recommend using the
[Revise](https://timholy.github.io/Revise.jl/stable/) package
which automatically detects changes you are making in your
packages and reflects them real-time in your current REPL session.
Using these strategies should minimize the number of times you
need to restart your REPL session.

!!! info "Coming soon"

    A guide to compiling the ITensors.jl package with PackageCompiler
    is coming soon.

## Multithreading Support

There are two possible sources of parallelization available in 
ITensors.jl, both external to the package right now. These are:
 - BLAS/LAPACK multithreading (through whatever flavor you are using, i.e. OpenBLAS or MKL).
 - The Strided.jl package, which implements a multithreaded array permutation.

The BLAS/LAPACK multithreading can be controlled in the usual way with 
environment variables, or within Julia. So for example, to control 
from Julia, you would do:
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
if you are using OpenBLAS, the command would be something like 
`ccall((:openblas_get_num_threads, Base.libblas_name), Cint, ())`.

Alternatively, you can use environment variables, so at your command 
line prompt you would use:
```
~ export MKL_NUM_THREADS=4
```
if you are using MKL or
```
~ export OPENBLAS_NUM_THREADS=4
```
if you are using OpenBLAS. We would highly recommend using MKL (see
the installation instructions for how to do that), especially if you 
are using an Intel chip. In general, we have not found MKL/OpenBLAS 
multithreading to help much in the context of common ITensor applications
(like DMRG), but your mileage may vary and it would depend highly on the 
problem you are studying. 
How well BLAS multithreading will work would depend on how much your 
calculations are dominated by matrix multiplications (which is not 
always the case, especially if you are using QN conservation).

Then, a separate level of mutlithreading could be turned on, which is 
native Julia multithreading. Right now in ITensors.jl, this would 
only control array permutation functions we use from 
[Strided.jl](https://github.com/Jutho/Strided.jl). You would set it 
with the environment variable `JULIA_NUM_THREADS`, for example:
```julia
julia> Threads.nthreads() # By default it is probably off
1
```
Then if you set `export JULIA_NUM_THREADS=4` at your command line, 
you would see the next time you start up Julia:
```julia
julia> Threads.nthreads()
4
```
As of this writing, we have not found that using that kind of 
multithreading has helped much in the context of DMRG calculation, 
but your mileage may vary. Also note that the two kinds of multithreading
(BLAS vs. native Julia) may compete with each other for resources,
so it is recommended you turn one or the other off.

We plan to incorporate our own multithreading with Julia's native 
multithreading capabilities, for example to parallelize over block 
sparse contractions. Stay tuned for that!

## Benchmarking and profiling

Julia has great built-in tools for benchmarking and profiling.
For benchmarking fast code at the command line, you can use
[BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/doc/manual.md):
```julia
julia> using ITensors;

julia> using BenchmarkTools;

julia> i = Index(100, "i");

julia> A = randomITensor(i, i');

julia> @btime 2*$A;
  4.279 μs (8 allocations: 78.73 KiB)
```

We recommend packages like [ProfileView](https://github.com/timholy/ProfileView.jl) 
to get detailed profiles of your code, in order to pinpoint functions 
or lines of code that are slower than they should be.

## ITensor type design and writing performant code

Advanced users might notice something strange about the definition
of the ITensor type, that it is often not "type stable". Some of 
this is by design. The definition for ITensor is:
```julia
mutable struct ITensor{N}
  inds::IndexSet{N}
  store::TensorStorage
end
```
These are both abstract types, which is something that is generally 
discouraged for peformance.

This has a few disadvantages. Some code that you might expect to be 
type stable, like `getindex`, is not, for example:
```julia
julia> i = Index(2, "i");

julia> A = randomITensor(i, i');

julia> @code_warntype A[i=>1, i'=>2]
Variables
  #self#::Core.Compiler.Const(getindex, false)
  T::ITensor{1}
  ivs::Tuple{Pair{Index{Int64},Int64}}
  p::Tuple{Union{Nothing, Int64}}
  vals::Tuple{Any}

Body::Number
1 ─ %1  = NDTensors.getperm::Core.Compiler.Const(NDTensors.getperm, false)
│   %2  = ITensors.inds(T)::IndexSet{1,IndexT,DataT} where DataT<:Tuple where IndexT<:Index
│   %3  = Base.broadcasted(ITensors.ind, ivs)::Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple},Nothing,typeof(ind),Tuple{Tuple{Pair{Index{Int64},Int64}}}}
│   %4  = Base.materialize(%3)::Tuple{Index{Int64}}
│         (p = (%1)(%2, %4))
│   %6  = NDTensors.permute::Core.Compiler.Const(NDTensors.permute, false)
│   %7  = Base.broadcasted(ITensors.val, ivs)::Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple},Nothing,typeof(val),Tuple{Tuple{Pair{Index{Int64},Int64}}}}
│   %8  = Base.materialize(%7)::Tuple{Int64}
│         (vals = (%6)(%8, p))
│   %10 = Core.tuple(T)::Tuple{ITensor{1}}
│   %11 = Core._apply_iterate(Base.iterate, Base.getindex, %10, vals)::Number
│   %12 = Core.typeassert(%11, ITensors.Number)::Number
└──       return %12

julia> typeof(A[i=>1, i'=>2])
Float64
```
Uh oh, that doesn't look good! Julia can't know ahead of time, based on 
the inputs, what the type of the output is, besides that it will be a
`Number` (though at runtime, the output has a concrete type, `Float64`).

So why is it designed this way? The main reason is to allow more 
generic and dynamic code than traditional, statically-typed Arrays.
This allows us to have code like:
```julia
A = randomITensor(i', i)
A .*= 2+1im
```
Here, the type of the storage of A is changed in-place (allocations
are performed only when needed).
More generally, this allows ITensors to have more generic in-place 
functionality, so you can write code where you don't know what the
storage is until runtime.

This can lead to certain types of code having perfomance problems, 
for example looping through ITensors with many elements can be slow:
```julia
julia> function myscale!(A::ITensor, x::Number)
         for n in 1:dim(A)
           A[n] = x * A[n]
         end
       end;

julia> d = 10_000;

julia> i = Index(d);

julia> @btime myscale!(A, 2) setup = (A = randomITensor(i));
  2.169 ms (117958 allocations: 3.48 MiB)
```
However, this is fast:
```
julia> function myscale!(A::Array, x::Number)
         for n in 1:length(A)
           A[n] = x * A[n]
         end
       end;

julia> @btime myscale!(A, 2) setup = (A = randn(d));
  3.451 μs (0 allocations: 0 bytes)

julia> myscale2!(A::ITensor, x::Number) = myscale!(array(A), x)
myscale2! (generic function with 1 method)

julia> @btime myscale2!(A, 2) setup = (A = randomITensor(i));
  3.571 μs (2 allocations: 112 bytes)
```
How does this work? It relies on a "function barrier" technique. 
Julia compiles functions "just-in-time", so that calls to an inner 
function written in terms of a type-stable type are still fast.
That inner function is compiled to very fast code.
The main overhead is that Julia has to determine which function 
to call at runtime.

Therefore, users should keep this in mind when they are writing 
ITensors.jl code, and we warn that explicitly looping over large 
ITensors by individual elements should be done with caution in 
performance critical sections of your code. 
However, be sure to benchmark and profile your code before 
prematurely optimizing, since you may end be surprised about 
what are the fast and slow parts of your code.

Some strategies for avoiding ITensor loops are:
 - Use broadcasting and other built-in ITensor functionality that makes use of function barriers.
 - Convert ITensors to type-stable collections like the Tensor type of NDTensors.jl and write functions in terms of the Tensor type (i.e. the function barrier techique that is used throughout ITensors.jl).
 - When initializing very large ITensors elementwise, use built-in ITensor constructors, or first construct an equivalent tensor as an Array or Tensor and then convert it to an ITensor.

## ITensor in-place operations

In-place operations can help with optimizing code, when the
memory of the output tensor of an operation is preallocated.

The main way to access this in ITensor is through broadcasting.
For example:
```julia
A = randomITensor(i, i')
B = randomITensor(i', i)
A .+= 2 .* B
```
Internally, this is rewritten by Julia as a call to `broadcast!`.
ITensors.jl overloads this call (or more specifically, a lower
level function `copyto!` written in terms of a special lazy type
that saves all of the objects and operations). Then, this call is 
rewritten as
```julia
map!((x,y) -> x+2*y, A, A, B)
```
This is mostly an optimization to use when you can preallocate
storage that can be used multiple times.

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
for convenience we might expand it in-place. If you are worried
about memory allocations, we recommend using benchmarking and
profiling to pinpoint slow parts of your code (often times, you
may be surprised by what is actually slow).

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
T[1,2,1] = 1.3  # Conventional element setting

i = Index(2)
T = Tensor(i,i',i')  # The identifiers are ignored, just interpreted as above
T[1,2,1] = 1.3
```
To make performant ITensor code (refer to the the previous section 
on type stability and function barriers), ITensor storage data and 
indices are passed by reference into Tensors, where the performance 
critical operations are performed.

An example of a function barrier using NDTensors is the following:
```julia
julia> using NDTensors

julia> d = 10_000;

julia> i = Index(d);

julia> function myscale!(A::Tensor, x::Number)
         for n in 1:dim(A)
           A[n] = x * A[n]
         end
       end;

julia> @btime myscale!(A, 2) setup = (A = Tensor(d));
  3.530 μs (0 allocations: 0 bytes)

julia> myscale2!(A::ITensor, x::Number) = myscale!(tensor(A), x)
myscale2! (generic function with 1 method)

julia> @btime myscale2!(A, 2) setup = (A = randomITensor(i));
  3.549 μs (2 allocations: 112 bytes)
```
A very efficient function is written for the Tensor type. Then,
the ITensor version just wraps the Tensor function by calling it
after converting the ITensor to a Tensor (without any copying)
with the `tensor` function.
This is the basis for the design of all performance critical ITensors.jl functions.
