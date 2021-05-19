# Index collections

Collections of `Index` are used throughout ITensors.jl to represent the dimensions of tensors. In general, collections that are recognized and returned by ITensors.jl functions are either `Vector` of `Index` or `Tuple` of `Index`, depending on the context. For example internally an `ITensor` has a static number of indices so stores a `Tuple` of `Index`, while set operations like `commoninds((i, j, k), (j, k, l))` will return a `Vector` `[j, k]` since the operation is inherently dynamic, i.e. the number of indices in the intersection can't in general be known before running the code. `Vector` of `Index` and `Tuple` of `Index` can usually be used interchangeably, but one or the other may be faster depending on the operation being performed.

## [Priming and tagging](@id Priming_and_tagging_IndexSet)

Documentation for priming and tagging collections of Index can be found in the ITensor [Priming and tagging](@ref Priming_and_tagging_ITensor) section.

## Set operations

Documentation for set operations involving Index collections can be found in the ITensor [Index collections set operations](@ref) section.

## Subsets

```@docs
getfirst(::Function, ::IndexSet)
getfirst(::IndexSet)
```

## Iterating

```@docs
eachval(::Index...)
eachindval(::Index...)
```


## Symmetry related properties

```@docs
dir(::IndexSet, ::Index)
```
