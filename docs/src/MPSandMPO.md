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
MPS(::Any)
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
prime!(::ITensors.AbstractMPS)
```

## Operations

```@docs
dag(::ITensors.AbstractMPS)
orthogonalize!
truncate!
replacebond!(::MPS, ::Int, ::ITensor; kwargs...)
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

