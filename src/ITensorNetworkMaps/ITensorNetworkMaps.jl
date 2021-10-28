module ITensorNetworkMaps

using ..ITensors
using LinearMaps

using ITensors: promote_itensor_eltype

import Base: *

import ITensors: contract

export ITensorNetworkMap, input_inds, output_inds

# convert from Tuple to Vector
tuple_to_vector(t::Tuple) = collect(t)
tuple_to_vector(v::Vector) = v

function default_input_inds(itensors::Vector{ITensor})
  return filter(i -> plev(i) == 0, noncommoninds(itensors...))
end

# Represents the action of applying the
# vector of ITensors to a starting state and then mapping
# them back (from output_inds to input_inds)
struct ITensorNetworkMap{T} <: LinearMap{T}
  itensors::Vector{ITensor}
  input_inds::Vector{Index}
  output_inds::Vector{Index}
  function ITensorNetworkMap(itensors::Vector{ITensor}, input_inds, output_inds)
    inds_in = tuple_to_vector(input_inds)
    inds_out = tuple_to_vector(output_inds)
    return new{promote_itensor_eltype(itensors)}(itensors, inds_in, inds_out)
  end
end
function ITensorNetworkMap(
  itensors::Vector{ITensor};
  input_inds=default_input_inds(itensors),
  output_inds=dag(input_inds'),
)
  return ITensorNetworkMap(itensors, input_inds, output_inds)
end

Base.size(T::ITensorNetworkMap) = (dim(output_inds(T)), dim(input_inds(T)))
Base.eltype(T::ITensorNetworkMap) = promote_itensor_eltype(T.itensors)
input_inds(T::ITensorNetworkMap) = T.input_inds
output_inds(T::ITensorNetworkMap) = T.output_inds

(T::ITensorNetworkMap * v::ITensor) = contract(pushfirst!(copy(T.itensors), v))
(T::ITensorNetworkMap)(v::ITensor) = replaceinds(T * v, output_inds(T) => input_inds(T))

(v::ITensor * T::LinearMap) = transpose(T) * v
contract(T::LinearMap) = T * ITensor(1)

function Base.transpose(T::ITensorNetworkMap)
  return ITensorNetworkMap(reverse(T.itensors), output_inds(T), input_inds(T))
end

# This is actually a Hermitian conjugation, not priming
function Base.adjoint(T::ITensorNetworkMap)
  return ITensorNetworkMap(
    reverse(dag.(T.itensors)), dag(output_inds(T)), dag(input_inds(T))
  )
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
  # TODO: Check it is actually an ITensorNetworkMap
  return input_inds(first(A.maps))
end
function output_inds(A::LinearMaps.CompositeMap)
  # TODO: Check it is actually an ITensorNetworkMap
  return output_inds(last(A.maps))
end

callable(x, y) = x(y)

apply(f, A::LinearMaps.ScaledMap, v::ITensor) = (A.λ * f(A.lmap, v))
(A::LinearMaps.ScaledMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMap * v::ITensor) = apply(*, A, v)

apply(f, A::LinearMaps.UniformScalingMap, v::ITensor) = (A.λ * v)
(A::LinearMaps.UniformScalingMap)(v::ITensor) = apply(callable, A, v)

function apply(f, A::LinearMaps.LinearCombination, v::ITensor)
  N = length(A.maps)
  Av = f(A.maps[1], v)
  for n in 2:N
    Av += f(A.maps[n], v)
  end
  return Av
end
(A::LinearMaps.LinearCombination)(v::ITensor) = apply(callable, A, v)

function _replaceinds(::typeof(callable), A::LinearMaps.CompositeMap, v::ITensor)
  output_inds_A = output_inds(A)
  input_inds_A = input_inds(A)
  return replaceinds(v, output_inds_A => input_inds_A)
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
