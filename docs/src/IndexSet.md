# IndexSet

```@docs
IndexSet(::Vector{<:Index})
```

## Priming and tagging methods

```@docs
prime(::IndexSet, ::Int)
```

```@docs
map(::Function, ::IndexSet)
```

## Set operations

```@docs
intersect(::IndexSet, ::IndexSet)
```

```@docs
firstintersect(::IndexSet, ::IndexSet)
```

```@docs
setdiff(::IndexSet, ::IndexSet)
```

```@docs
firstsetdiff(::IndexSet, ::IndexSet)
```

## Subsets

```@docs
getfirst(::Function, ::IndexSet)
getfirst(::IndexSet)
```

```@docs
filter(::Function, ::IndexSet)
```
