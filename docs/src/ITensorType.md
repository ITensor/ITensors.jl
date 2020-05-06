# ITensor

## Description

```@docs
ITensor
```

## Dense Constructors

```@docs
ITensor(::Type{<:Number}, ::IndexSet)
ITensor(::Type{<:Number}, ::UndefInitializer, ::IndexSet)
randomITensor(::Type{<:Number}, ::IndexSet)
setelt(::IndexVal)
```

## QN BlockSparse Constructors

```@docs
ITensor(::Type{<:Number}, ::QN, ::IndexSet)
ITensor(::Type{<:Number}, ::ITensors.QNIndexSet)
```

## Zero Constructors

```@docs
zeroITensor(::Type{<:Number}, ::IndexSet)
```

## QN Zero Constructors

```@docs
zeroITensor(::Type{<:Number}, ::ITensors.QNIndexSet)
```

## Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::IndexSet)
diagITensor(::Vector{<:Number}, ::IndexSet)
diagITensor(::Number, ::IndexSet)
delta(::Type{<:Number}, ::IndexSet)
```

## QN Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::QN, ::IndexSet)
delta(::Type{<:Number}, ::QN, ::IndexSet)
```

## Getting and setting elements

```@docs
getindex(::ITensor, ::Any...)
getindex(::ITensor{N}, ::Vararg{Int,N}) where {N}
setindex!(::ITensor, ::Number, ::Any...)
setindex!(::ITensor, ::Number, ::Int...)
```

## Properties

```@docs
inds(::ITensor)
ind(::ITensor, ::Int)
```

## Priming and tagging

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
```

## Index Manipulations

```@docs
replaceind(::ITensor, ::Any...)
replaceinds(::ITensor, ::Any...)
swapind(::ITensor, ::Any...)
swapinds(::ITensor, ::Any...)
```

## Math operations

```@docs
*(::ITensor, ::ITensor)
exp(::ITensor, ::Any)
```

## Decompositions
```@docs
svd(::ITensor, ::Any...)
factorize(::ITensor, ::Any...)
```

## Operations

```@docs
permute(::ITensor, ::Any)
```

