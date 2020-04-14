# ITensor

```@docs
ITensor
```

## Constructors

```@docs
ITensor(::IndexSet)
```

```@docs
ITensor(::Number, ::Index...)
```

```@docs
randomITensor(::Type{<:Number}, ::IndexSet)
```

## Sparse constructors

```@docs
diagITensor(::IndexSet)
```

```@docs
delta(::Type{<:Number}, ::IndexSet)
```

## Operations

```@docs
*(::ITensor, ::ITensor)
```

```@docs
exp(::ITensor, ::Any)
```

## Decompositions
```@docs
svd(::ITensor, ::Any...)
```

