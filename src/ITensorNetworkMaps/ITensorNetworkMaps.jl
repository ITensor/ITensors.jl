module ITensorNetworkMaps

using ..ITensors
using LinearMaps

import Base: *

export ITensorNetworkMap, input_inds, output_inds

############################################################################
abstract type AbstractITensorMap end

input_inds(T::AbstractITensorMap) = T.input_inds
output_inds(T::AbstractITensorMap) = T.output_inds
(T::AbstractITensorMap)(v::ITensor) = replaceinds(T * v, output_inds(T) => input_inds(T))

# convert from Tuple to Vector
tuple_to_vector(t::Tuple) = collect(t)
tuple_to_vector(v::Vector) = v

# Represents the action of applying the
# vector of ITensors to a starting state and then mapping
# them back (from output_inds to input_inds)
# TODO: rename ITensorNetworkMap?
# TODO: maybe parametrize the type to allow storing just 1 ITensor?
# TODO: contraction order optimization!
struct ITensorMap <: AbstractITensorMap
  itensors::Vector{ITensor}
  scalar::Number
  input_inds::Vector{Index}
  output_inds::Vector{Index}
  function ITensorMap(itensors::Vector{ITensor}, scalar, input_inds, output_inds)
    inds_in = tuple_to_vector(input_inds)
    inds_out = tuple_to_vector(output_inds)
    #inds_eltype = promote_type(eltype(input_inds), eltype(output_inds))
    #return new{inds_eltype}(itensors, inds_in, inds_out)
    return new(itensors, scalar, inds_in, inds_out)
  end
end
Base.size(T::AbstractITensorMap) = (dim(output_inds(T)), dim(input_inds(T)))
(T::ITensorMap * v::ITensor) = T.scalar * contract(pushfirst!(copy(T.itensors), v))
function Base.transpose(T::ITensorMap)
  return ITensorMap(reverse(T.itensors), output_inds(T), input_inds(T))
end

# This is actually a Hermitian conjugation, not priming
function Base.adjoint(T::ITensorMap)
  return ITensorMap(reverse(dag.(T.itensors)), dag(output_inds(T)), dag(input_inds(T)))
end

function ITensorMap(itensors::Vector{ITensor}, input_inds, output_inds)
  return ITensorMap(itensors, true, input_inds, output_inds)
end

function default_input_inds(itensors::Vector{ITensor})
  return filter(i -> plev(i) == 0, noncommoninds(itensors...))
end

function ITensorMap(
  itensors::Vector{ITensor};
  input_inds=default_input_inds(itensors),
  output_inds=dag(input_inds'),
)
  return ITensorMap(itensors, input_inds, output_inds)
end
############################################################################

struct ITensorNetworkMap{T} <: LinearMap{T}
  A::ITensorMap
end
ITensorNetworkMap(A::ITensorMap) = ITensorNetworkMap{eltype(A)}(A)
ITensorNetworkMap(tn::Vector{ITensor}; kwargs...) = ITensorNetworkMap(ITensorMap(tn; kwargs...))

Base.size(A::ITensorNetworkMap) = size(A.A)

function input_inds(A::ITensorNetworkMap)
  return input_inds(A.A)
end
function output_inds(A::ITensorNetworkMap)
  return output_inds(A.A)
end

function input_inds(A::LinearMaps.LinearCombination)
  in_inds = input_inds(A.maps[1])
  @assert all(M -> hassameinds(input_inds(M), in_inds), A.maps)
  return in_inds
end
function output_inds(A::LinearMaps.LinearCombination)
  out_inds = output_inds(A.maps[1])
  @assert all(M -> hassameinds(output_inds(M), out_inds), A.maps)
  return out_inds
end

function input_inds(A::LinearMaps.CompositeMap)
  # TODO: it is actually an ITensorNetworkMap
  return input_inds(A.maps[1])
end
function output_inds(A::LinearMaps.CompositeMap)
  # TODO: it is actually an ITensorNetworkMap
  return output_inds(A.maps[end])
end

LinearAlgebra.adjoint(A::ITensorNetworkMap) = ITensorNetworkMap(adjoint(A.A))
LinearAlgebra.transpose(A::ITensorNetworkMap) = ITensorNetworkMap(transpose(A.A))

callable(x, y) = x(y)

function apply(f, A::ITensorMap, v::ITensor)
  return f(A, v)
end

# Application on ITensor
apply(f, A::ITensorNetworkMap, v::ITensor) = f(A.A, v)
(A::ITensorNetworkMap)(v::ITensor) = apply(callable, A, v)
(A::ITensorNetworkMap * v::ITensor) = apply(*, A, v)

apply(f, A::LinearMaps.ScaledMap, v::ITensor) = (A.λ * f(A.lmap, v))
(A::LinearMaps.ScaledMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.ScaledMap * v::ITensor) = apply(*, A, v)

apply(f, A::LinearMaps.UniformScalingMap, v::ITensor) = (A.λ * v)
(A::LinearMaps.UniformScalingMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.UniformScalingMap * v::ITensor) = apply(*, A, v)

function apply(f, A::LinearMaps.LinearCombination, v::ITensor)
  N = length(A.maps)
  Av = f(A.maps[1], v)
  for n in 2:N
    Av += f(A.maps[n], v)
  end
  return Av
end
(A::LinearMaps.LinearCombination)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.LinearCombination * v::ITensor) = apply(*, A, v)

function _replaceinds(::typeof(callable), A::LinearMaps.CompositeMap, v::ITensor)
  return replaceinds(v, output_inds(A.maps[end]) => input_inds(A.maps[1]))
end
function _replaceinds(::typeof(*), A::LinearMaps.CompositeMap, v::ITensor)
  return v
end

function apply(f, A::LinearMaps.CompositeMap, v::ITensor)
  N = length(A.maps)
  Av = v
  for n in 1:N
    Av = A.maps[n] * Av
  end
  Av = _replaceinds(f, A, Av)
  return Av
end
(A::LinearMaps.CompositeMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.CompositeMap * v::ITensor) = apply(*, A, v)

function apply(f, A::LinearMaps.BlockMap, v::Vector{ITensor})
  nrows = length(A.rows)
  ncols = A.rows[1]
  @assert all(==(ncols), A.rows)
  M = reshape(collect(A.maps), nrows, ncols)
  Av = fill(ITensor(), nrows)
  for i in 1:nrows, j in 1:ncols
    Av[i] += f(M[i, j], v[j])
  end
  return Av
end
(A::LinearMaps.BlockMap)(v::Vector{ITensor}) = apply(callable, A, v)
(A::LinearMaps.BlockMap * v::Vector{ITensor}) = apply(*, A, v)

end
