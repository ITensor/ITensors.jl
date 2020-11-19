
# Advanced ITensor usage guide

## Installing and updating ITensors.jl

The ITensors package can be installed with the Julia package manager.
Assuming you have already downloaded Julia, which you can get
[here](https://julialang.org/downloads/), from the Julia REPL, 
type `]` to enter the Pkg REPL mode and run:
```
$ julia
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
$ julia
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

A common place you might accidentally come across this is when
you are creating a Hamiltonian with `AutoMPO`:
```julia
julia> N = 4;

julia> sites = siteinds("S=1/2",N);

julia> ampo = AutoMPO();

julia> for j=1:N-1
         ampo += "Sz", j, "Sz", j+1
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

## Finding documentation interactively

Julia provides many tools for searching for documentation interactively at the REPL. Say that you want to learn more about how to use an ITensor from the command line. You can start by typing `?` followed by `ITensor`:
```julia
julia> using ITensors

julia> ?ITensor
search: ITensor ITensors itensor emptyITensor randomITensor

  An ITensor is a tensor whose interface is independent of its
  memory layout. Therefore it is not necessary to know the ordering
  of an ITensor's indices, only which indices an ITensor has.
  Operations like contraction and addition of ITensors automatically
  handle any memory permutations.

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  julia> i = Index(2, "i")
  (dim=2|id=287|"i")
  
  julia> A = randomITensor(i', i)
  ITensor ord=2 (dim=2|id=287|"i")' (dim=2|id=287|"i")
  NDTensors.Dense{Float64,Array{Float64,1}}
  
  julia> @show A;
  A = ITensor ord=2
  Dim 1: (dim=2|id=287|"i")'
  Dim 2: (dim=2|id=287|"i")
  NDTensors.Dense{Float64,Array{Float64,1}}
   2×2
   0.28358594718392427   1.4342219756446355
   1.6620103556283987   -0.40952231269251566
  
  julia> @show inds(A);
  inds(A) = IndexSet{2} (dim=2|id=287|"i")' (dim=2|id=287|"i") 
[...]
```
(the specific output may be different for different versions of ITensors.jl as we update the docs). You can use the help prompt (which you get by typing `?` at the REPL) to print out documentation for types and methods.

Another way to get information about types is with the function `fieldnames`:
```julia
julia> fieldnames(ITensor)
(:store, :inds)
```
which shows the fields of a type. Note that in general the specific names of the fields and structures of types may change (we consider those to be internal details), however we often make functions to access the fields of a type that have the same name as the field, so it is a good place to get started. For example, you can access the storage and indices of an ITensor `A` with the functions `store(A)` and `inds(A)`.

Another helpful function is `apropos`, which search through all documentation for a string (ignoring the case) and prints a list of all types and methods with documentation that contain the string. For example:
```julia
julia> apropos("IndexSet")
ITensors.IndexSet
ITensors.push
ITensors.insertat
ITensors.getfirst
ITensors.commoninds
ITensors.pushfirst
NDTensors.mindim
[...]
```
This can often return too much information. A helpful way to narrow down the search is with regular expressions, for example:
```julia
julia> apropos(r"ITensor.*IndexSet")
ITensors.block
ITensors.hasinds
ITensors.ITensor
NDTensors.inds
```
where the notation `r"..."` is Julia notation for making a string that will be interpreted as a [regular expression](https://docs.julialang.org/en/v1/manual/strings/#Regular-Expressions). Here, we are searching for any documentation that contains the string "ITensor" followed at some point by "IndexSet". The notation `.*` is regular expression notation for matching any number of any type of character.

Based on the `apropos` function, we can make some helper functions that may be useful. For example:
```julia
using ITensors

function finddocs(s)
  io = IOBuffer()
  apropos(io, s)
  v = chomp(String(take!(io)))
  return split(v, "\n")
end

function finddocs(s...)
  intersect(finddocs.(s)...)
end

found_methods = finddocs("indices", "set difference")
display(found_methods)
```
returns:
```julia
3-element Array{SubString{String},1}:
 "ITensors.noncommoninds"
 "Base.setdiff"
 "ITensors.uniqueinds"
```
which are the functions that have docs that contain the strings `"indices"` and `"set difference"`. We can print the docs for `uniqueinds` to find:
```julia
help?> uniqueinds
search: uniqueinds unique_siteinds uniqueind uniqueindex

  uniqueinds(A, B; kwargs...)
  uniqueinds(::Order{N}, A, B; kwargs...)


  Return an IndexSet with indices that are unique to the set of
  indices of A and not in B (the set difference).

  Optionally, specify the desired number of indices as Order(N),
  which adds a check and can be a bit more efficient.
```

We can also filter the results to only specify functions from certain modules, for example:
```julia
julia> filter(x -> startswith(x, "ITensors"), finddocs("indices", "set difference"))
2-element Array{SubString{String},1}:
 "ITensors.noncommoninds"
 "ITensors.uniqueinds"

julia> filter(x -> !startswith(x, "ITensors"), finddocs("indices", "set difference"))
1-element Array{SubString{String},1}:
 "Base.setdiff"
```
Ideally we could have `apropos` do a "smart" Google-like search of the appropriate docstrings, but this is a pretty good start.

Additionally, the `names` function can be useful, which prints the names of all functions and types that are exported by a module. For example:
```julia
julia> names(ITensors)
264-element Array{Symbol,1}:
 Symbol("@OpName_str")
 Symbol("@SiteType_str")
 Symbol("@StateName_str")
 Symbol("@TagType_str")
 Symbol("@disable_warn_order")
 Symbol("@reset_warn_order")
 Symbol("@set_warn_order")
 Symbol("@ts_str")
 :AbstractObserver
 :AutoMPO
 :DMRGObserver
 :ITensor
 :ITensors
 :Index
[...]
```
Of course this is a very long list (and the methods are returned as `Symbol`s, which are like strings but not as easy to work with). However, we can convert the list to strings and filter the strings to find functions we are interested in, for example:
```julia
julia> filter(x -> contains(x, "common") && contains(x, "ind"), String.(names(ITensors)))
8-element Array{String,1}:
 "common_siteind"
 "common_siteinds"
 "commonind"
 "commonindex"
 "commoninds"
 "hascommoninds"
 "noncommonind"
 "noncommoninds"
```

Julia types do not have member functions, so people coming from object oriented programming languages may find that at first it is more difficult to find methods that are applicable to a certain type. However, Julia has many fantastic tools for introspection that we can use to make this task easier.

## Make a small project based on ITensors.jl

Once you start to have longer code, you will want to put your
code into one or more files. For example, you may have a short script
with one or more functions based on ITensors.jl:
```julia
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

As your code gets longer, you can split it into multiple files
and `include` this files into one main project file, for example
if you have two files with functions in them:
```julia
# file1.jl

function norm2(A::ITensor)
  return (A*dag(A))[]
end
```
and
```julia
# file2.jl

function square(A::ITensor)
  return A .^ 2
end
```

```julia
# my_itensor_project.jl

using ITensors

include("file1.jl")

include("file2.jl")
```
Then, as before, you can use your functions at the Julia REPL
by just including the file `my_itensor_project.jl`:
```julia
julia> include("my_itensor_project.jl");

julia> i = Index(2; tags="i");

julia> A = randomITensor(i', i);

julia> norm2(A)
[...]

julia> square(A)
[...]
```

As your code gets more complicated and has more files, it is helpful
to organize it into a package. That will be covered in the
next section.

## Make a Julia package based on ITensors.jl

In this section, we will describe how to make a Julia package based on
ITensors.jl. This is useful to do when your project gets longer,
since it helps with:
 - Code organization.
 - Adding dependencies that will get automatically installed through Julia's package system.
 - Versioning.
 - Automated testing.
 - Code sharing and easier package installation.
 - Officially registering your package with Julia.
and many more features that we will mention later.

Start up Julia and install [PkgTemplates](https://invenia.github.io/PkgTemplates.jl/stable/)
```julia
$ julia

julia> ]

pkg> add PkgTemplates
```
then press backspace and type:
```
julia> using PkgTemplates

julia> t = Template(; user="your_github_username", plugins=[Git(; ssh=true),])

julia> t("MyITensorsPkg")
```
You should put your Github account name instead of `"your_github_username"`,
if you want to use Github to host your package. 
The option `plugins=[Git(; ssh=true),]` sets the Github authentication to use
ssh, which is generally more convenient. You can switch to https (where you
have to type your username and password to push changes) by setting `ssh=false`
or leaving off `plugins=[...]`. By default, the package will be located in
the directory `~/.julia/dev`, you can change this with the keyword argument
`dir=[...]`. However, `~/.julia/dev` is recommended since that is the directory
Julia's package manager (and other packages like `Revise`) will look for development
packages. Please see the `PkgTemplate` documentation for more customization options.

Then, we want to tell Julia about our new package. We do this as
follows:
```julia
julia> ]

pkg> dev ~/.julia/dev/MyITensorsPkg
```
then you can do:
```julia
julia> using MyITensorsPkg
```
from any directory to use your new package. However, it doesn't 
have any functions available yet. Additionally, there should be
an empty test file already set up here:
```
~/.julia/dev/MyITensorsPkg/test/runtests.jl
```
which you can run from any directory like:
```julia
julia> ]

pkg> test MyITensorsPkg
```
It should show something like:
```julia
[...]
Test Summary:    |
MyITensorsPkg.jl | No tests
    Testing MyITensorsPkg tests passed 
```
since there are no tests yet.

First we want to add ITensors as a dependency of our package.
We do this by "activating" our package environment and then
adding ITensors:
```julia
julia> ]

pkg> activate MyITensorsPkg

(MyITensorsPkg) pkg> add ITensors
```
This will edit the file `~/.julia/dev/MyITensorsPkg/Project.toml`
and add the line
```
[deps]
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"
```
Because your package is under development, back in the main
Pkg environment you should type `resolve`:
```julia
(MyITensorsPkg) pkg> activate

pkg> resolve
```
Now, if you or someone else uses the package, it will automatically
install ITensors.jl for you.

Now your package is set up to develop! Try editing the file
`~/.julia/dev/MyITensorsPkg/src/MyITensorsPkg.jl` and add the 
`norm2` function, which calculates the squared norm of an ITensor:
```julia
module MyITensorsPkg

using ITensors

export norm2

norm2(A::ITensor) = (A*dag(A))[]

end
```
The export command makes `norm2` available in the namespace without
needing to type `MyITensorsPkg.norm2` when you do 
`using MyITensorsPkg`. Now in a new Julia session you can do:
```julia
julia> using ITensors

julia> i = Index(2)
(dim=2|id=263)

julia> A = randomITensor(i)
ITensor ord=1 (dim=2|id=263)
NDTensors.Dense{Float64,Array{Float64,1}}

julia> norm(A)^2
6.884457016011188

julia> norm2(A)
ERROR: UndefVarError: norm2 not defined
[...]

julia> using MyITensorsPkg

julia> norm2(A)
6.884457016011188
```
Unfortunately, if you continue to edit the file `MyITensorsPkg.jl`,
even if you type `using MyITensorsPkg` again, if you are in the
same Julia session the changes will not be reflected, and
you will have to restart your Julia session. The 
[Revise](https://timholy.github.io/Revise.jl/stable/) package
will allow you to edit your package files and have the changes
reflected in real time in your current Julia session, so you
don't have to restart the session.

Now, we can add some tests for our new functionality. Edit
the file `~/.julia/dev/MyITensorsPkg/test/runtests.jl` to
look like:
```julia
using MyITensorsPkg
using ITensors
using Test

@testset "MyITensorsPkg.jl" begin
  i = Index(2)
  A = randomITensor(i)
  @test isapprox(norm2(A), norm(A)^2)
end
```
Now when you test your package you should see:
```julia
pkg> test MyITensorsPkg
[...]
Test Summary:    | Pass  Total
MyITensorsPkg.jl |    1      1
    Testing MyITensorsPkg tests passed 
```

Your package should already be set up as a git repository by 
the `PkgTemplates` commands we started with.
We recommend using Github or similar versions control systems
for your packages, especially if you plan to make them public
and officially register them as Julia packages.

You can set up your local package as a Github repository by
following the steps [here](https://help.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line). Many of the steps may be unnecessary since they
were already set up by `PkgTemplates`. You should be able to
go to the website [here](https://github.com/new), create a new
Github repository with the name `MyITensorsPkg.jl`, and then following
the instructions under "push an existing repository from the command line".

You may also want to switch between HTTPS and SSH authentication
as described [here](https://help.github.com/en/github/using-git/changing-a-remotes-url),
if you didn't choose your preferred authentication protocol with
PkgTemplates.

There are many more features you can add to your package through 
various Julia packages and Github, for example:
 - Control of precompilation with tools like [SnoopCompile](https://timholy.github.io/SnoopCompile.jl/stable/).
 - Automatic testing of your package at every pull request/commit with Github Actions, Travis, or similar services.
 - Automated benchmarking of your package at every pull request with [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl), [PkgBenchmark](https://juliaci.github.io/PkgBenchmark.jl/stable/) and [BenchmarkCI](https://github.com/tkf/BenchmarkCI.jl).
 - Automated building of your documentation with [Documenter](https://juliadocs.github.io/Documenter.jl/stable/).
 - Compiling your package with [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/).
 - Automatically check what parts of your code your tests check with code coverage.
 - Officially register your Julia package so that others can easily install it and follow along with updated versions using the [Registrator](https://juliaregistries.github.io/Registrator.jl/stable/).
You can take a look at the [ITensors](https://github.com/ITensor/ITensors.jl) 
Github page for inspiration on setting up some of these services
and ideas for organizing your package.

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
 - Staying in the same Julia session with [Revise](https://timholy.github.io/Revise.jl/stable/).
 - Using [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/) to compile ITensors.jl ahead of time.

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

If you plan to use ITensors.jl directly from the command line
(i.e. not from the REPL), and the startup time is an issue,
you can try compiling ITensors.jl using [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/).

Before using PackageCompiler to compile ITensors, when we first start using ITensors.jl we might see:
```julia
julia> @time using ITensors
  3.845253 seconds (10.96 M allocations: 618.071 MiB, 3.95% gc time)

julia> @time i = Index(2);
  0.000684 seconds (23 allocations: 20.328 KiB)

julia> @time A = randomITensor(i', i);
  0.071022 seconds (183.24 k allocations: 9.715 MiB)

julia> @time svd(A, i');
  5.802053 seconds (24.56 M allocations: 1.200 GiB, 7.83% gc time)

julia> @time svd(A, i');
  0.000177 seconds (450 allocations: 36.609 KiB)
```
ITensors provides the command `ITensors.compile()` to create what is
called a "custom system image", a custom version of Julia that
includes a compiled version of ITensors (see the [PackageCompiler documentation](https://julialang.github.io/PackageCompiler.jl/dev/) for more details).
Just run the command:
```
julia> ITensors.compile()
[...]
```
By default, this will create the file `sys_itensors.so` in the directory
`~/.julia/sysimages`.
Then if we start julia with:
```
$ julia --sysimage ~/.julia/sysimages/sys_itensors.so
```
then you should see something like:
```julia
julia> @time using ITensors
  0.330587 seconds (977.61 k allocations: 45.807 MiB, 1.89% gc time)

julia> @time i = Index(2);
  0.000656 seconds (23 allocations: 20.328 KiB)

julia> @time A = randomITensor(i', i);
  0.000007 seconds (7 allocations: 576 bytes)

julia> @time svd(A, i');
  0.263526 seconds (290.02 k allocations: 14.220 MiB)

julia> @time svd(A, i');
  0.000135 seconds (350 allocations: 29.984 KiB)
```
which is much better. 

Note that you will have to recompile ITensors with the command 
`ITensors.compile()` any time that you update the version of ITensors
in order to keep the system image updated. We hope to make this
process more automated in the future.

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
$ export MKL_NUM_THREADS=4
```
if you are using MKL or
```
$ export OPENBLAS_NUM_THREADS=4
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
julia> i = Index(2, "i")
(dim=2|id=811|"i")

julia> A = emptyITensor(i', i);

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=811|"i")'
Dim 2: (dim=2|id=811|"i")
NDTensors.Empty{Float64,NDTensors.Dense{Float64,Array{Float64,1}}}
 2×2



julia> A[i' => 1, i => 2] = 1.2;

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=811|"i")'
Dim 2: (dim=2|id=811|"i")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 0.0  1.2
 0.0  0.0
```
Here, the type of the storage of A is changed in-place. It starts as an `Empty` storage, a special trivial storage. When we set an element, we then allocate the appropriate storage. Allocations are performed only when needed, so if another element is set then no allocation is performed.
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
prematurely optimizing, since you may be surprised about 
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
T = Tensor((i,i',i'))  # The identifiers are ignored, just interpreted as above
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
