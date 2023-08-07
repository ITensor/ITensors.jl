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
function randomITensor(rng::AbstractRNG, ::Type{S}, is::Indices; kwargs...) where {S<:Number}
  v = randn(rng, S, dim(is))
  return ITensor(AllowAlias(), S, v, is; kwargs...)
end

function randomITensor(::Type{S}, is::Indices) where {S<:Number}
  return randomITensor(Random.default_rng(), S, is)
end

function randomITensor(::Type{S}, is...) where {S<:Number}
  return randomITensor(Random.default_rng(), S, is...)
end

function randomITensor(rng::AbstractRNG, ::Type{S}, is...) where {S<:Number}
  return randomITensor(rng, S, indices(is...))
end

randomITensor(is::Indices) = randomITensor(Random.default_rng(), is)
randomITensor(rng::AbstractRNG, is::Indices) = randomITensor(rng, NDTensors.default_eltype(), is)
randomITensor(is...) = randomITensor(Random.default_rng(), is...)
randomITensor(rng::AbstractRNG, is...) = randomITensor(rng, NDTensors.default_eltype(), indices(is...))