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
random_mps(sites::Vector{<:Index})
random_mps(::Type{<:Number}, sites::Vector{<:Index})
random_mps(::Vector{<:Index}, ::Any)
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
copy(::ITensorMPS.AbstractMPS)
deepcopy(::ITensorMPS.AbstractMPS)
```

## Properties

```@docs
eltype(::ITensorMPS.AbstractMPS)
flux(::ITensorMPS.AbstractMPS)
hasqns(::ITensorMPS.AbstractMPS)
length(::ITensorMPS.AbstractMPS)
maxlinkdim(::ITensorMPS.AbstractMPS)
```

## Obtaining and finding indices

```@docs
siteinds(::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS, ::Int)
siteinds(::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS, ::Int)
findsite
findsites
firstsiteinds
linkind(::ITensorMPS.AbstractMPS,::Int)
siteind(::MPS, ::Int)
siteind(::typeof(first), ::MPS, ::Int)
siteinds(::MPS)
siteind(::MPO, ::Int)
siteinds(::MPO)
siteinds(::ITensorMPS.AbstractMPS, ::Int)
```

## Priming and tagging

```@docs
prime(::ITensorMPS.AbstractMPS)
prime(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
prime(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
prime(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
prime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)

swapprime(::ITensorMPS.AbstractMPS, args...; kwargs...)

setprime(::ITensorMPS.AbstractMPS)
setprime(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
setprime(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
setprime(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
setprime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)

noprime(::ITensorMPS.AbstractMPS)
noprime(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
noprime(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
noprime(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
noprime(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)

addtags(::ITensorMPS.AbstractMPS)
addtags(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
addtags(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
addtags(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
addtags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)

removetags(::ITensorMPS.AbstractMPS)
removetags(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
removetags(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
removetags(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
removetags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)

replacetags(::ITensorMPS.AbstractMPS)
replacetags(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
replacetags(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
replacetags(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
replacetags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)

settags(::ITensorMPS.AbstractMPS)
settags(::typeof(siteinds), ::ITensorMPS.AbstractMPS)
settags(::typeof(linkinds), ::ITensorMPS.AbstractMPS)
settags(::typeof(siteinds), ::typeof(commoninds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
settags(::typeof(siteinds), ::typeof(uniqueinds), ::ITensorMPS.AbstractMPS, ::ITensorMPS.AbstractMPS)
```

## Operations

```@docs
expect(::MPS, ::Any)
correlation_matrix(::MPS, ::AbstractString, ::AbstractString)
dag(::ITensorMPS.AbstractMPS)
dense(::ITensorMPS.AbstractMPS)
movesite(::ITensorMPS.AbstractMPS, ::Pair{Int, Int};orthocenter::Int,kwargs...)
orthogonalize!
replacebond!(::MPS, ::Int, ::ITensor)
sample(::MPS)
sample!(::MPS)
sample(::MPO)
swapbondsites(::ITensorMPS.AbstractMPS, ::Int; kwargs...)
truncate!
```

## Gate evolution

```@docs
product(::ITensor, ::ITensorMPS.AbstractMPS)
product(::Vector{ITensor}, ::ITensorMPS.AbstractMPS)
```

## Algebra Operations

```@docs
inner(::MPST, ::MPST) where {MPST <: ITensorMPS.AbstractMPS}
dot(::MPST, ::MPST) where {MPST <: ITensorMPS.AbstractMPS}
loginner(::MPST, ::MPST) where {MPST <: ITensorMPS.AbstractMPS}
logdot(::MPST, ::MPST) where {MPST <: ITensorMPS.AbstractMPS}
inner(::MPS, ::MPO, ::MPS)
dot(::MPS, ::MPO, ::MPS)
inner(::MPO, ::MPS, ::MPO, ::MPS)
dot(::MPO, ::MPS, ::MPO, ::MPS)
norm(::ITensorMPS.AbstractMPS)
normalize(::ITensorMPS.AbstractMPS)
normalize!(::ITensorMPS.AbstractMPS)
lognorm(::ITensorMPS.AbstractMPS)
+(::ITensorMPS.AbstractMPS...)
contract(::MPO, ::MPS)
apply(::MPO, ::MPS)
contract(::MPO, ::MPO)
apply(::MPO, ::MPO)
error_contract(y::MPS, A::MPO, x::MPS)
outer(::MPS, ::MPS)
projector(::MPS)
```

