"""
    randomITensor([::Type{ElT <: Number} = Float64, ]inds)
    randomITensor([::Type{ElT <: Number} = Float64, ]inds::Index...)

Construct an ITensor with type `ElT` and indices `inds`, whose elements are
normally distributed random numbers. If the element type is not specified,
it defaults to `Float64`.

# Examples

```julia
i = Index(2,"index_i")
j = Index(4,"index_j")
k = Index(3,"index_k")

A = randomITensor(i,j)
B = randomITensor(ComplexF64,undef,k,j)
```
"""
function randomITensor(::Type{S}, is::Indices) where {S<:Number}
  return randomITensor(Random.default_rng(), S, is)
end

function randomITensor(rng::AbstractRNG, ::Type{S}, is::Indices) where {S<:Number}
  T = ITensor(S, undef, is)
  randn!(rng, T)
  return T
end

function randomITensor(::Type{S}, is...) where {S<:Number}
  return randomITensor(Random.default_rng(), S, is...)
end

function randomITensor(rng::AbstractRNG, ::Type{S}, is...) where {S<:Number}
  return randomITensor(rng, S, indices(is...))
end

# To fix ambiguity with QN version
function randomITensor(::Type{ElT}, is::Tuple{}) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, is)
end

# To fix ambiguity with QN version
function randomITensor(rng::AbstractRNG, ::Type{ElT}, is::Tuple{}) where {ElT<:Number}
  return randomITensor(rng, ElT, Index{Int}[])
end

# To fix ambiguity with QN version
function randomITensor(is::Tuple{})
  return randomITensor(Random.default_rng(), is)
end

# To fix ambiguity with QN version
function randomITensor(rng::AbstractRNG, is::Tuple{})
  return randomITensor(rng, Float64, is)
end

# To fix ambiguity errors with QN version
function randomITensor(::Type{ElT}) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT)
end

# To fix ambiguity errors with QN version
function randomITensor(rng::AbstractRNG, ::Type{ElT}) where {ElT<:Number}
  return randomITensor(rng, ElT, ())
end

randomITensor(is::Indices) = randomITensor(Random.default_rng(), is)
randomITensor(rng::AbstractRNG, is::Indices) = randomITensor(rng, Float64, is)
randomITensor(is...) = randomITensor(Random.default_rng(), is...)
randomITensor(rng::AbstractRNG, is...) = randomITensor(rng, Float64, indices(is...))

# To fix ambiguity errors with QN version
randomITensor() = randomITensor(Random.default_rng())

# To fix ambiguity errors with QN version
randomITensor(rng::AbstractRNG) = randomITensor(rng, Float64, ())
