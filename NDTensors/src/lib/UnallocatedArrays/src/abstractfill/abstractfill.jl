using FillArrays: AbstractFill
using NDTensors.SetParameters: Position, get_parameter, set_parameters
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
