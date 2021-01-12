This shows an example of multithreading block sparse contraction.

First start julia with some number of threads, for example:
```
$ julia -t 4
```
starts Julia with 4 threads.

Then, include the file and run the `main` function:
```julia
julia> include("main.jl")

julia> main(d = 4, order = 4)
```
This contracts two order-4 tensors over 4 indices, with blocks of dimension 4x4x4x4.

