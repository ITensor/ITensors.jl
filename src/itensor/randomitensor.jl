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
function randomITensor(
  rng::AbstractRNG, ::Type{S}, is::Indices; kwargs...
) where {S<:Number}
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
function randomITensor(rng::AbstractRNG, is::Indices)
  return randomITensor(rng, NDTensors.default_eltype(), is)
end
randomITensor(is...) = randomITensor(Random.default_rng(), is...)
function randomITensor(rng::AbstractRNG, is...)
  return randomITensor(rng, NDTensors.default_eltype(), indices(is...))
end

### From QN ITensor
"""
    randomITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds)
    randomITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::Index...)

Construct an ITensor with `NDTensors.BlockSparse` storage filled with random
elements of type `ElT` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`. If the flux is not specified it defaults to `QN()`.
"""
function randomITensor(::Type{ElT}, flux::QN, inds::Indices) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, flux, inds)
end

function randomITensor(
  rng::AbstractRNG, ::Type{ElT}, flux::QN, inds::Indices
) where {ElT<:Number}
  T = ITensor(ElT, undef, flux, inds)
  randn!(rng, T)
  return T
end

function randomITensor(::Type{ElT}, flux::QN, is...) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, flux, is...)
end

function randomITensor(rng::AbstractRNG, ::Type{ElT}, flux::QN, is...) where {ElT<:Number}
  return randomITensor(rng, ElT, flux, indices(is...))
end

function randomITensor(::Type{ElT}, inds::QNIndices) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, inds)
end

function randomITensor(rng::AbstractRNG, ::Type{ElT}, inds::QNIndices) where {ElT<:Number}
  return randomITensor(rng, ElT, QN(), inds)
end

function randomITensor(flux::QN, inds::Indices)
  return randomITensor(Random.default_rng(), flux, inds)
end

function randomITensor(rng::AbstractRNG, flux::QN, inds::Indices)
  return randomITensor(rng, Float64, flux, inds)
end

function randomITensor(flux::QN, is...)
  return randomITensor(Random.default_rng(), flux, is...)
end

function randomITensor(rng::AbstractRNG, flux::QN, is...)
  return randomITensor(rng, Float64, flux, indices(is...))
end

# TODO: generalize to list of Tuple, Vector, and QNIndex
function randomITensor(::Type{ElT}, inds::QNIndex...) where {ElT<:Number}
  return randomITensor(Random.default_rng(), ElT, inds...)
end

# TODO: generalize to list of Tuple, Vector, and QNIndex
function randomITensor(rng::AbstractRNG, ::Type{ElT}, inds::QNIndex...) where {ElT<:Number}
  return randomITensor(rng, ElT, QN(), inds)
end

randomITensor(inds::QNIndices) = randomITensor(Random.default_rng(), inds)

randomITensor(rng::AbstractRNG, inds::QNIndices) = randomITensor(rng, Float64, QN(), inds)

# TODO: generalize to list of Tuple, Vector, and QNIndex
randomITensor(inds::QNIndex...) = randomITensor(Random.default_rng(), inds...)

# TODO: generalize to list of Tuple, Vector, and QNIndex
randomITensor(rng::AbstractRNG, inds::QNIndex...) = randomITensor(rng, Float64, QN(), inds)
