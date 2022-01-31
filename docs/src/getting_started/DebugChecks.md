# Enabling Debug Checks

ITensor provides some optional checks for common errors, which we call "debug checks".
These can be enabled with the command:
```julia
ITensors.enable_debug_checks()
```
and disabled with the command:
```julia
ITensors.disable_debug_checks()
```

We recommend enabling debug checks when you are developing and testing your code, and then
disabling them when running in production to get the best performance.

For example, when debug checks are turned on, ITensor checks that all indices of an ITensor
are unique (if they are not unique, it leads to undefined behavior in tensor operations
like contraction, addition, and decomposition):
```julia
julia> using ITensors

julia> i = Index(2)
(dim=2|id=913)

julia> A = randomITensor(i', i)
ITensor ord=2 (dim=2|id=913)' (dim=2|id=913)
NDTensors.Dense{Float64, Vector{Float64}}

julia> noprime(A)
ITensor ord=2 (dim=2|id=913) (dim=2|id=913)
NDTensors.Dense{Float64, Vector{Float64}}

julia> ITensors.enable_debug_checks()
using_debug_checks (generic function with 1 method)

julia> noprime(A)
ERROR: Trying to create ITensors with collection of indices ((dim=2|id=913), (dim=2|id=913)). Indices must be unique.
Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:33
 [2] macro expansion
   @ ~/.julia/packages/ITensors/cu9Bo/src/itensor.jl:85 [inlined]
 [3] macro expansion
   @ ~/.julia/packages/ITensors/cu9Bo/src/global_variables.jl:177 [inlined]
 [4] ITensor
   @ ~/.julia/packages/ITensors/cu9Bo/src/itensor.jl:82 [inlined]
 [5] #itensor#123
   @ ~/.julia/packages/ITensors/cu9Bo/src/itensor.jl:123 [inlined]
 [6] itensor(args::NDTensors.DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, NDTensors.Dense{Float64, Vector{Float64}}})
   @ ITensors ~/.julia/packages/ITensors/cu9Bo/src/itensor.jl:123
 [7] noprime(::ITensor; kwargs::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
   @ ITensors ~/.julia/packages/ITensors/cu9Bo/src/itensor.jl:1211
 [8] noprime(::ITensor)
   @ ITensors ~/.julia/packages/ITensors/cu9Bo/src/itensor.jl:1211
 [9] top-level scope
   @ REPL[7]:1
```
You can track where debug checks are located in the code [here](https://github.com/ITensor/ITensors.jl/search?q=debug_check),
and add your own debug checks to your own code by wrapping your code with the macro `ITensors.@debug_check`.
