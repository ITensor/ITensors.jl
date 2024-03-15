using BlockArrays: BlockedUnitRange
# TODO: Implement these.
function blocksortperm end
function blockmergesortperm(::BlockedUnitRange)
  return error("Not implemented yet.")
end
function dual end
function fuse end
function invblockperm end
function sector end

# Represents the range `1:1` or `Base.OneTo(1)`.
struct OneToOne{T} <: AbstractUnitRange{T} end
OneToOne() = OneToOne{Bool}()
Base.first(a::OneToOne) = one(eltype(a))
Base.last(a::OneToOne) = one(eltype(a))

# https://github.com/ITensor/ITensors.jl/blob/v0.3.57/NDTensors/src/lib/GradedAxes/src/tensor_product.jl
# https://en.wikipedia.org/wiki/Tensor_product
# https://github.com/KeitaNakamura/Tensorial.jl
function tensor_product(
  a1::AbstractUnitRange,
  a2::AbstractUnitRange,
  a3::AbstractUnitRange,
  a_rest::Vararg{AbstractUnitRange},
)
  return foldl(tensor_product, (a1, a2, a3, a_rest...))
end

function tensor_product(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return error("Not implemented yet.")
end

function tensor_product(a1::Base.OneTo, a2::Base.OneTo)
  return Base.OneTo(length(a1) * length(a2))
end

function tensor_product(a1::OneToOne, a2::AbstractUnitRange)
  return a2
end

function tensor_product(a1::AbstractUnitRange, a2::OneToOne)
  return a1
end

function tensor_product(a1::OneToOne, a2::OneToOne)
  return OneToOne()
end

using BlockArrays: blockedrange, blocks
function tensor_product(a1::BlockedUnitRange, a2::BlockedUnitRange)
  return blockedrange(prod.(length, vec(collect(Iterators.product(blocks.((a1, a2))...)))))
end
