# TagSets.jl

A sorted collection of unique tags of type `T`.

# TODO

- Add `skipchars` (see `skipmissing`) and `delim` for delimiter.
- https://docs.julialang.org/en/v1/base/strings/#Base.strip
- https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/#Delimited-Files
- Add a `Bool` param for bounds checking/ignoring overflow/spillover?
- Make `S` a first argument, hardcode `SmallVector` storage?
- https://juliacollections.github.io/DataStructures.jl/v0.9/sorted_containers.html
- https://github.com/JeffreySarnoff/SortingNetworks.jl
- https://github.com/vvjn/MergeSorted.jl
- https://bkamins.github.io/julialang/2023/08/25/infiltrate.html
- https://github.com/Jutho/TensorKit.jl/blob/master/src/auxiliary/dicts.jl
- https://github.com/tpapp/SortedVectors.jl
- https://discourse.julialang.org/t/special-purpose-subtypes-of-arrays/20327
- https://discourse.julialang.org/t/all-the-ways-to-group-reduce-sorted-vectors-ideas/45239
- https://discourse.julialang.org/t/sorting-a-vector-of-fixed-size/71766
