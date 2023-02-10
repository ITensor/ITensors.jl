# ITensor automated benchmarks

These benchmarks run for every PR. They compare a set of benchmarks (such as basic ITensor operations like contraction) between the PR and the main branch of ITensor.

To run the benchmarks, you should go to the benchmark folder and activate the benchmarks project, and run the benchmarks:
```julia
julia> using ITensors

julia> cd(joinpath(pkgdir(ITensors), "benchmark"))

julia> using Pkg

julia> Pkg.activate(".")

julia> include("benchmarks.jl")

julia> SUITE
[...]
```
The benchmark results will be stored in the "SUITE" object.

Then you can run the benchmark suite using the [interface from BenchmarkTools](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/#Tuning-and-running-a-BenchmarkGroup).
```julia
tune!(SUITE);
results = run(SUITE; verbose=true, seconds=1)
```

Alternatively, you can run the benchmarks with the [BencharmkCI interface](https://github.com/tkf/BenchmarkCI.jl#running-benchmarkci-interactively=).
```julia
cd(pkgdir(ITensors))
using BenchmarkCI
BenchmarkCI.judge()
BenchmarkCI.displayjudgement()
```

## Development

If you are developing the ITensors.jl or NDTensors.jl packages, and any of the dependencies of either of those get updated, you will need to update the Manifest.toml file.

You can do that by removing the Manifest.toml file, checking out the ITensors and NDTensors modules for development, and then create a new Manifest.toml file with `Pkg.resolve`:
```julia
julia> using ITensors

julia> cd(joinpath(pkgdir(ITensors), "benchmark"))

julia> rm("Manifest.toml"; force=true)

julia> using Pkg

julia> Pkg.activate(".")

julia> Pkg.develop(path=joinpath("..", "NDTensors")) # Develop NDTensors

julia> Pkg.develop(path="..") # Develop ITensors

julia> Pkg.resolve()
```
