# MPS and MPO

## Types

```@docs
MPS
MPO
```

## MPS Constructors

```@docs
MPS(::Int)
MPS(::Type{<:Number}, ::Any)
randomMPS
productMPS
```

## MPO Constructors

```@docs
MPO(::Int)
MPO(::Any, ::Vector{String})
MPO(::Any, ::String)
```

## Properties

```@docs
length(::ITensors.AbstractMPS)
maxlinkdim(::ITensors.AbstractMPS)
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
orthogonalize!
truncate!
replacebond!(::MPS, ::Int, ::ITensor)
sample(::MPS)
sample!(::MPS)
```

## Algebra Operations

```@docs
dot(::MPS, ::MPS)
+(::MPS, ::MPS)
+(::MPO, ::MPO)
*(::MPO, ::MPS)
```

