using BlockArrays: BlockedUnitRange, blocks

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
