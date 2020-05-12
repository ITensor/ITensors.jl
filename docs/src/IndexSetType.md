# IndexSet

```@docs
IndexSet(::Vector{<:Index})
```

## Priming and tagging methods

```@docs
prime(::IndexSet, ::Int)
map(::Function, ::IndexSet)
```

## Set operations

```@docs
intersect(::IndexSet, ::IndexSet)
firstintersect(::IndexSet, ::IndexSet)
setdiff(::IndexSet, ::IndexSet)
firstsetdiff(::IndexSet, ::IndexSet)
```

## Subsets

```@docs
getfirst(::Function, ::IndexSet)
getfirst(::IndexSet)
filter(::Function, ::IndexSet)
```
