using LinearAlgebra: mul!, norm

sparse_norm(a::AbstractArray, p::Real=2) = norm(sparse_storage(a))

function default_mul!!(a_dest::AbstractMatrix, a1::AbstractMatrix, a2::AbstractMatrix, α::Number=true, β::Number=false)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

function default_mul!!(a_dest::Number, a1::Number, a2::Number, α::Number=true, β::Number=false)
  return a1 * a2 * α + a_dest * β
end

# a1 * a2 * α + a_dest * β
# Assumes that `a_dest` has been zeroed out
# already.
function sparse_mul_zeroed!(
  a_dest::AbstractMatrix,
  a1::AbstractMatrix,
  a2::AbstractMatrix,
  α::Number=true,
  β::Number=false;
  (mul!!)=(default_mul!!),
)
  for I1 in stored_indices(a1)
    for I2 in stored_indices(a2)
      if Tuple(I1)[2] == Tuple(I2)[1]
        I_dest = CartesianIndex(Tuple(I1)[1], Tuple(I2)[2])
        a_dest[I_dest] = mul!!(a_dest[I_dest], a1[I1], a2[I2], α, β)
      end
    end
  end
  return a_dest
end

# a1 * a2 * α + a_dest * β
function sparse_mul!(
  a_dest::AbstractMatrix,
  a1::AbstractMatrix,
  a2::AbstractMatrix,
  α::Number=true,
  β::Number=false;
  (mul!!)=(default_mul!!),
)
  zerovector!(a_dest)
  sparse_mul_zeroed!(a_dest, a1, a2, α, β; mul!!)
  return a_dest
end
