# IndexSet

```@docs
IndexSet(::Vector{<:Index})
```

## Priming and tagging methods

```@docs
prime(::ITensor, ::Any...)
setprime(::ITensor, ::Any...)
noprime(::ITensor, ::Any...)
mapprime(::ITensor, ::Any...)
swapprime(::ITensor, ::Any...)
addtags(::ITensor, ::Any...)
removetags(::ITensor, ::Any...)
replacetags(::ITensor, ::Any...)
settags(::ITensor, ::Any...)
swaptags(::ITensor, ::Any...)
map(::Function, ::IndexSet)
```

## Set operations

```@docs
commoninds
hascommoninds
uniqueinds
noncommoninds
unioninds
```

## Subsets

```@docs
getfirst(::Function, ::IndexSet)
getfirst(::IndexSet)
filter(::Function, ::IndexSet)
```
