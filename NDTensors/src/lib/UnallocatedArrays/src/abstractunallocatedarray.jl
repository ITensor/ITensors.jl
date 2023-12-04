@inline Base.axes(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = axes(parent(A))
Base.size(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = size(parent(A))
function FillArrays.getindex_value(A::Union{<:UnallocatedFill,<:UnallocatedZeros})
  return getindex_value(parent(A))
end

function Base.complex(A::Union{<:UnallocatedFill,<:UnallocatedZeros})
  return set_alloctype(
    complex(parent(A)), set_parameters(alloctype(A), Position{1}(), complex(eltype(A)))
  )
end

# mult_fill(a, b, val, ax) = Fill(val, ax)
# mult_zeros(a, b, elt, ax) = Zeros{elt}(ax)
# mult_ones(a, b, elt, ax) = Ones{elt}(ax)

# broadcasted_fill(f, a, val, ax) = Fill(val, ax)
# broadcasted_fill(f, a, b, val, ax) = Fill(val, ax)
# broadcasted_zeros(f, a, elt, ax) = Zeros{elt}(ax)
# broadcasted_zeros(f, a, b, elt, ax) = Zeros{elt}(ax)
# broadcasted_ones(f, a, elt, ax) = Ones{elt}(ax)
# broadcasted_ones(f, a, b, elt, ax) = Ones{elt}(ax)

# kron_fill(a, b, val, ax) = Fill(val, ax)
# kron_zeros(a, b, elt, ax) = Zeros{elt}(ax)
# kron_ones(a, b, elt, ax) = Ones{elt}(ax)
