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

function Base.transpose(a::Union{<:UnallocatedFill, <:UnallocatedZeros})
  return set_alloctype(transpose(parent(a)), alloctype(a))
end

function Base.adjoint(a::Union{<:UnallocatedFill, <:UnallocatedZeros})
  return set_alloctype(adjoint(parent(a)), alloctype(a))
end

# mult_fill(a, b, val, ax) = Fill(val, ax)
# mult_ones(a, b, elt, ax) = Ones{elt}(ax)

# broadcasted_fill(f, a, val, ax) = Fill(val, ax)
# broadcasted_fill(f, a, b, val, ax) = Fill(val, ax)
# broadcasted_ones(f, a, elt, ax) = Ones{elt}(ax)
# broadcasted_ones(f, a, b, elt, ax) = Ones{elt}(ax)

# kron_fill(a, b, val, ax) = Fill(val, ax)
# kron_ones(a, b, elt, ax) = Ones{elt}(ax)
