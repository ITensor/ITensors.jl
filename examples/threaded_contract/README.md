This shows an example of multithreading block sparse contraction.

First start Julia with some number of threads, for example:
```
$ julia -t 4
```
starts Julia with 4 threads.

Then, include the file and run the `main` function:
```julia
julia> include("main.jl")

julia> main(d = 20, order = 4)
#################################################
# order = 4
# d = 20
#################################################

Serial contract:
  21.465 ms (131 allocations: 7.34 MiB)

Threaded contract:
  7.933 ms (445 allocations: 7.37 MiB)

C_contract ≈ C_threaded_contract = true
```
This contracts two order-4 tensors over 4 indices, with blocks of dimension 20x20x20x20.

Note that because of the overhead involved in threading the block sparse contractions, contracting many small blocks may not benefit (or can be worse) when multithreading is enabled.

