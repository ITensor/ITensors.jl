using BlockArrays: BlockedUnitRange, blocks

# Represents the range `1:1` or `Base.OneTo(1)`.
struct OneToOne{T} <: AbstractUnitRange{T} end
OneToOne() = OneToOne{Bool}()
Base.first(a::OneToOne) = one(eltype(a))
Base.last(a::OneToOne) = one(eltype(a))

# https://en.wikipedia.org/wiki/Tensor_product
# https://github.com/KeitaNakamura/Tensorial.jl
tensor_product(a1, a2, a3, as...) = foldl(tensor_product, (a1, a2, a3, as...))
tensor_product(a1, a2) = error("Not implemented for $(typeof(a1)) and $(typeof(a2)).")

function tensor_product(a1::Base.OneTo, a2::Base.OneTo)
  return Base.OneTo(length(a1) * length(a2))
end

function tensor_product(a1::BlockedUnitRange, a2::BlockedUnitRange)
  return blockedrange(prod.(length, vec(collect(Iterators.product(blocks.((a1, a2))...)))))
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
