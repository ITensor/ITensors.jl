# ITensor

## Description

```@docs
ITensor
```

## Dense Constructors

```@docs
ITensor(::Type{<:Number}, ::ITensors.Indices)
ITensor(::Type{<:Number}, ::UndefInitializer, ::ITensors.Indices)
randomITensor(::Type{<:Number}, ::ITensors.Indices)
setelt(::IndexVal)
```

## QN BlockSparse Constructors

```@docs
ITensor(::Type{<:Number}, ::QN, ::ITensors.Indices)
ITensor(A::Array, inds::ITensors.QNIndexSet)
```

## Empty Constructors

```@docs
emptyITensor(::Type{<:Number}, ::ITensors.Indices)
```

## QN Empty Constructors

```@docs
emptyITensor(::Type{<:Number}, ::ITensors.QNIndices)
```

## Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::ITensors.Indices)
diagITensor(::Vector{<:Number}, ::ITensors.Indices)
diagITensor(::Number, ::ITensors.Indices)
delta(::Type{<:Number}, ::ITensors.Indices)
```

## QN Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::QN, ::ITensors.Indices)
delta(::Type{<:Number}, ::QN, ::ITensors.Indices)
```

## Convert to Array

```@docs
Array{ElT, N}(::ITensor{N}, ::Vararg{Index, N}) where {ElT, N}
array(::ITensor)
matrix(::ITensor{2})
vector(::ITensor{1})
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

## [Priming and tagging](@id Priming_and_tagging_ITensor)

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

## IndexSet set operations

```@docs
commoninds
commonind
uniqueinds
uniqueind
noncommoninds
noncommonind
unioninds
unionind
hascommoninds
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
exp(::ITensor, ::Any, ::Any)
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

