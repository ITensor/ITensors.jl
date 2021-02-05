# MPS and MPO

## Types

```@docs
MPS
MPO
```

## MPS Constructors

```@docs
MPS(::Int)
MPS(::Type{<:Number}, ::Vector{<:Index})
randomMPS(sites::Vector{<:Index}; linkdim=1)
randomMPS(::Type{<:Number}, sites::Vector{<:Index}; linkdim=1)
randomMPS(sites::Vector{<:Index}, state; linkdim=1)
productMPS(::Vector{<:Index},states)
productMPS(::Type{<:Number},::Vector{<:Index},states)
productMPS(::Vector{<:IndexVal})
productMPS(::Type{<:Number}, ::Vector{<:IndexVal})
```

## MPO Constructors

```@docs
MPO(::Int)
MPO(::Type{<:Number}, ::Vector{<:Index}, ::Vector{String})
MPO(::Type{<:Number}, ::Vector{<:Index}, ::String)
MPO(::MPS)
```

## Properties

```@docs
length(::ITensors.AbstractMPS)
flux(::ITensors.AbstractMPS)
maxlinkdim(::ITensors.AbstractMPS)
hasqns(::ITensors.AbstractMPS)
```

## Obtaining and finding indices

```@docs
siteinds(::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS, ::Int)
siteinds(::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS, ::Int)
findsite
findsites
firstsiteinds
linkind(::ITensors.AbstractMPS,::Int)
siteind(::MPS, ::Int)
siteind(::typeof(first), ::MPS, ::Int)
siteinds(::MPS)
siteind(::MPO, ::Int)
siteinds(::MPO)
siteinds(::ITensors.AbstractMPS, ::Int)
```

## Priming and tagging

```@docs
prime(::ITensors.AbstractMPS)
prime!(::ITensors.AbstractMPS)
prime(::typeof(linkinds), ::ITensors.AbstractMPS)
prime!(::typeof(linkinds), ::ITensors.AbstractMPS)
prime(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
prime!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
prime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
prime!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
setprime(::ITensors.AbstractMPS)
setprime!(::ITensors.AbstractMPS)
setprime(::typeof(linkinds), ::ITensors.AbstractMPS)
setprime!(::typeof(linkinds), ::ITensors.AbstractMPS)
setprime(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
setprime!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
setprime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
setprime!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
noprime(::ITensors.AbstractMPS)
noprime!(::ITensors.AbstractMPS)
noprime(::typeof(linkinds), ::ITensors.AbstractMPS)
noprime!(::typeof(linkinds), ::ITensors.AbstractMPS)
noprime(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
noprime!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
noprime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
noprime!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
addtags(::ITensors.AbstractMPS)
addtags!(::ITensors.AbstractMPS)
addtags(::typeof(linkinds), ::ITensors.AbstractMPS)
addtags!(::typeof(linkinds), ::ITensors.AbstractMPS)
addtags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
addtags!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
addtags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
addtags!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
removetags(::ITensors.AbstractMPS)
removetags!(::ITensors.AbstractMPS)
removetags(::typeof(linkinds), ::ITensors.AbstractMPS)
removetags!(::typeof(linkinds), ::ITensors.AbstractMPS)
removetags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
removetags!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
removetags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
removetags!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
replacetags(::ITensors.AbstractMPS)
replacetags!(::ITensors.AbstractMPS)
replacetags(::typeof(linkinds), ::ITensors.AbstractMPS)
replacetags!(::typeof(linkinds), ::ITensors.AbstractMPS)
replacetags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
replacetags!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
replacetags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
replacetags!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
settags(::ITensors.AbstractMPS)
settags!(::ITensors.AbstractMPS)
settags(::typeof(linkinds), ::ITensors.AbstractMPS)
settags!(::typeof(linkinds), ::ITensors.AbstractMPS)
settags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
settags!(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
settags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
settags!(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
```

## Operations

```@docs
dag(::ITensors.AbstractMPS)
dag!(::ITensors.AbstractMPS)
dense(::ITensors.AbstractMPS)
movesite(::ITensors.AbstractMPS, ::Pair{Int, Int};orthocenter::Int,kwargs...)
orthogonalize!
replacebond!(::MPS, ::Int, ::ITensor)
sample(::MPS)
sample!(::MPS)
sample(::MPO)
swapbondsites(::ITensors.AbstractMPS, ::Int; kwargs...)
truncate!
```

## Gate evolution

```@docs
product(::Vector{ <: ITensor}, ::ITensors.AbstractMPS)
```

## Algebra Operations

```@docs
dot(::MPST, ::MPST) where {MPST <: ITensors.AbstractMPS}
logdot(::MPST, ::MPST) where {MPST <: ITensors.AbstractMPS}
norm(::ITensors.AbstractMPS)
lognorm(::ITensors.AbstractMPS)
+(::MPS, ::MPS)
*(::MPO, ::MPS)
```

