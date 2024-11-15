using LinearAlgebra: dot, mul!, norm

sparse_norm(a::AbstractArray, p::Real=2) = norm(sparse_storage(a))

function mul_indices(I1::CartesianIndex{2}, I2::CartesianIndex{2})
  if I1[2] ≠ I2[1]
    return nothing
  end
  return CartesianIndex(I1[1], I2[2])
end

# TODO: Is this needed? Maybe when multiplying vectors?
function mul_indices(I1::CartesianIndex{1}, I2::CartesianIndex{1})
  if I1 ≠ I2
    return nothing
  end
  return CartesianIndex(I1)
end

function default_mul!!(
  a_dest::AbstractMatrix,
  a1::AbstractMatrix,
  a2::AbstractMatrix,
  α::Number=true,
  β::Number=false,
)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

function default_mul!!(
  a_dest::Number, a1::Number, a2::Number, α::Number=true, β::Number=false
)
  return a1 * a2 * α + a_dest * β
end

# a1 * a2 * α + a_dest * β
function sparse_mul!(
  a_dest::AbstractArray,
  a1::AbstractArray,
  a2::AbstractArray,
  α::Number=true,
  β::Number=false;
  (mul!!)=(default_mul!!),
)
  for I1 in stored_indices(a1)
    for I2 in stored_indices(a2)
      I_dest = mul_indices(I1, I2)
      if !isnothing(I_dest)
        a_dest[I_dest] = mul!!(a_dest[I_dest], a1[I1], a2[I2], α, β)
      end
    end
  end
  return a_dest
end

function sparse_dot(a1::AbstractArray, a2::AbstractArray)
  # This requires that `a1` and `a2` have the same shape.
  # TODO: Generalize (Base supports dot products of
  # arrays with the same length but different sizes).
  size(a1) == size(a2) ||
    throw(DimensionMismatch("Sizes $(size(a1)) and $(size(a2)) don't match."))
  dot_dest = zero(Base.promote_op(dot, eltype(a1), eltype(a2)))
  # TODO: First check if the number of stored elements (`stored_length`, to be renamed
  # `stored_length`) is smaller in `a1` or `a2` and use whicheven one is smallar
  # as the outer loop.
  for I1 in stored_indices(a1)
    # TODO: Overload and use `Base.isstored(a, I) = I in stored_indices(a)` instead.
    # TODO: This assumes fast lookup of indices, which may not always be the case.
    # It could be better to loop over `stored_indices(a2)` and check that
    # `I1 == I2` instead (say using `mul_indices(I1, I2)`. We could have a trait
    # `HasFastIsStored(a::AbstractArray)` to choose between the two.
    if I1 in stored_indices(a2)
      dot_dest += dot(a1[I1], a2[I1])
    end
  end
  return dot_dest
end
