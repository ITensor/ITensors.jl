# ITensor

## Description

```@docs
ITensor
```

## Dense Constructors

```@docs
ITensor(::Type{<:Number}, ::ITensors.Indices)
ITensor(::Type{<:Number}, ::UndefInitializer, ::ITensors.Indices)
ITensor(::Type{ElT}, x::Number, inds::ITensors.Indices) where {ElT<:Number}
ITensor(as::ITensors.AliasStyle, ::Type{ElT}, A::Array{<:Number}, inds::ITensors.Indices; kwargs...) where {ElT<:Number}
randomITensor(::Type{<:Number}, ::ITensors.Indices)
onehot
```

## Dense View Constructors

```@docs
itensor(::Array{<:Number},::ITensors.Indices)
```

## QN BlockSparse Constructors

```@docs
ITensor(::Type{<:Number}, ::QN, ::ITensors.Indices)
ITensor(::ITensors.AliasStyle, ::Type{ElT}, A::Array{<:Number}, inds::ITensors.QNIndices; tol=0) where {ElT<:Number}
ITensor(::Type{<:Number}, ::UndefInitializer, ::QN, ::ITensors.Indices)
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
diagITensor(::Type{ElT}, is::ITensors.Indices) where {ElT}
diagITensor(as::ITensors.AliasStyle, ::Type{ElT}, v::Vector{<:Number}, is...) where {ElT<:Number}
diagITensor(as::ITensors.AliasStyle, ::Type{ElT}, x::Number, is...) where {ElT<:Number}
delta(::Type{<:Number}, ::ITensors.Indices)
```

## QN Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::QN, ::ITensors.Indices)
delta(::Type{<:Number}, ::QN, ::ITensors.Indices)
```

## Convert to Array

```@docs
Array{ElT, N}(::ITensor, ::Vararg{Index, N}) where {ElT, N}
array(::ITensor)
matrix(::ITensor)
vector(::ITensor)
```

## Getting and setting elements

```@docs
getindex(::ITensor, ::Any...)
setindex!(::ITensor, ::Number, ::Int...)
```

## Properties

```@docs
inds(::ITensor)
ind(::ITensor, ::Int)
dir(::ITensor, ::Index)
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

## Index collections set operations

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
eigen(::ITensor, ::Any, ::Any)
factorize(::ITensor, ::Any...)
```

## Memory operations

```@docs
permute(::ITensor, ::Any)
```

