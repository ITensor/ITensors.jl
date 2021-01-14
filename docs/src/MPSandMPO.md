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
common_siteind(::ITensors.AbstractMPS, ::ITensors.AbstractMPS, ::Int)
common_siteinds(::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
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
unique_siteind(::ITensors.AbstractMPS, ::ITensors.AbstractMPS, ::Int)
unique_siteinds(::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
```

## Priming and tagging

```@docs
prime(::ITensors.AbstractMPS)
prime!(::ITensors.AbstractMPS)
setprime(::ITensors.AbstractMPS)
setprime!(::ITensors.AbstractMPS)
noprime(::ITensors.AbstractMPS)
noprime!(::ITensors.AbstractMPS)
addtags(::ITensors.AbstractMPS)
addtags!(::ITensors.AbstractMPS)
removetags(::ITensors.AbstractMPS)
removetags!(::ITensors.AbstractMPS)
replacetags(::ITensors.AbstractMPS)
replacetags!(::ITensors.AbstractMPS)
settags(::ITensors.AbstractMPS)
settags!(::ITensors.AbstractMPS)
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

