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
randomMPS(sites::Vector{<:Index})
randomMPS(::Type{<:Number}, sites::Vector{<:Index})
randomMPS(::Vector{<:Index}, ::Any)
MPS(::Vector{<:Index}, ::Any)
MPS(::Type{<:Number}, ::Vector{<:Index}, ::Any)
MPS(::Vector{<:Pair{<:Index}})
MPS(::Type{<:Number}, ::Vector{<:Pair{<:Index}})
```

## MPO Constructors

```@docs
MPO(::Int)
MPO(::Type{<:Number}, ::Vector{<:Index}, ::Vector{String})
MPO(::Type{<:Number}, ::Vector{<:Index}, ::String)
```

## Copying behavior

```@docs
copy(::ITensors.AbstractMPS)
deepcopy(::ITensors.AbstractMPS)
```

## Properties

```@docs
eltype(::ITensors.AbstractMPS)
flux(::ITensors.AbstractMPS)
hasqns(::ITensors.AbstractMPS)
length(::ITensors.AbstractMPS)
maxlinkdim(::ITensors.AbstractMPS)
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
prime(::typeof(siteinds), ::ITensors.AbstractMPS)
prime(::typeof(linkinds), ::ITensors.AbstractMPS)
prime(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
prime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)

swapprime(::ITensors.AbstractMPS, args...; kwargs...)

setprime(::ITensors.AbstractMPS)
setprime(::typeof(siteinds), ::ITensors.AbstractMPS)
setprime(::typeof(linkinds), ::ITensors.AbstractMPS)
setprime(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
setprime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)

noprime(::ITensors.AbstractMPS)
noprime(::typeof(siteinds), ::ITensors.AbstractMPS)
noprime(::typeof(linkinds), ::ITensors.AbstractMPS)
noprime(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
noprime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)

addtags(::ITensors.AbstractMPS)
addtags(::typeof(siteinds), ::ITensors.AbstractMPS)
addtags(::typeof(linkinds), ::ITensors.AbstractMPS)
addtags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
addtags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)

removetags(::ITensors.AbstractMPS)
removetags(::typeof(siteinds), ::ITensors.AbstractMPS)
removetags(::typeof(linkinds), ::ITensors.AbstractMPS)
removetags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
removetags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)

replacetags(::ITensors.AbstractMPS)
replacetags(::typeof(siteinds), ::ITensors.AbstractMPS)
replacetags(::typeof(linkinds), ::ITensors.AbstractMPS)
replacetags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
replacetags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)

settags(::ITensors.AbstractMPS)
settags(::typeof(siteinds), ::ITensors.AbstractMPS)
settags(::typeof(linkinds), ::ITensors.AbstractMPS)
settags(::typeof(siteinds), ::typeof(commoninds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
settags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensors.AbstractMPS, ::ITensors.AbstractMPS)
```

## Operations

```@docs
expect(::MPS, ::Any)
correlation_matrix(::MPS, ::AbstractString, ::AbstractString)
dag(::ITensors.AbstractMPS)
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
product(::ITensor, ::ITensors.AbstractMPS)
product(::Vector{ITensor}, ::ITensors.AbstractMPS)
```

## Algebra Operations

```@docs
inner(::MPST, ::MPST) where {MPST <: ITensors.AbstractMPS}
dot(::MPST, ::MPST) where {MPST <: ITensors.AbstractMPS}
loginner(::MPST, ::MPST) where {MPST <: ITensors.AbstractMPS}
logdot(::MPST, ::MPST) where {MPST <: ITensors.AbstractMPS}
inner(::MPS, ::MPO, ::MPS)
dot(::MPS, ::MPO, ::MPS)
inner(::MPO, ::MPS, ::MPO, ::MPS)
dot(::MPO, ::MPS, ::MPO, ::MPS)
norm(::ITensors.AbstractMPS)
normalize(::ITensors.AbstractMPS)
normalize!(::ITensors.AbstractMPS)
lognorm(::ITensors.AbstractMPS)
+(::ITensors.AbstractMPS...)
contract(::MPO, ::MPS)
apply(::MPO, ::MPS)
contract(::MPO, ::MPO)
apply(::MPO, ::MPO)
error_contract(y::MPS, A::MPO, x::MPS)
outer(::MPS, ::MPS)
projector(::MPS)
```

