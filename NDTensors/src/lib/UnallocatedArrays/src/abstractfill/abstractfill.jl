## Here are functions specifically defined for UnallocatedArrays
## not implemented by FillArrays
## TODO determine min number of functions needed to be forwarded
alloctype(A::AbstractFill) = alloctype(typeof(A))
function alloctype(Atype::Type{<:AbstractFill})
  return get_parameter(Atype, Position{4}())
end

allocate(A::AbstractFill) = alloctype(A)(parent(A))

set_eltype(T::Type{<:AbstractFill}, elt::Type) = set_parameters(T, Position{1}(), elt)
set_ndims(T::Type{<:AbstractFill}, n) = set_parameters(T, Position{2}(), n)
set_axes(T::Type{<:AbstractFill}, ax::Type) = set_parameters(T, Position{3}(), ax)

## TODO get these working for UnallocatedX
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
