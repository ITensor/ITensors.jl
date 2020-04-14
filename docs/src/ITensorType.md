# ITensor

```@docs
ITensor
```

## Constructors

```@docs
ITensor(::IndexSet)
ITensor(::Number, ::Index...)
randomITensor(::Type{<:Number}, ::IndexSet)
```

## Sparse constructors

```@docs
diagITensor(::IndexSet)
delta(::Type{<:Number}, ::IndexSet)
```

## Operations

```@docs
*(::ITensor, ::ITensor)
exp(::ITensor, ::Any)
```

## Decompositions
```@docs
svd(::ITensor, ::Any...)
```

