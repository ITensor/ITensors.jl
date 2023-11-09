## Here are functions specifically defined for UnallocatedArrays
  ## not implemented by FillArrays
  ## TODO determine min number of functions needed to be forwarded

alloctype(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = alloctype(typeof(A))
alloctype(Atype::Type{<:Union{<:UnallocatedFill, <:UnallocatedZeros}}) = get_parameter(Atype, Position{4}())

allocate(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = alloctype(A)(parent(A))

## TODO Still working here I am not sure these functions and the
## Set parameter functions are working properly
set_alloctype(F::Type{<:Union{<:UnallocatedFill,<:UnallocatedZeros}}, alloc::Type{<:AbstractArray}) =
  set_parameter(F, Position{4}(), alloc)
set_eltype(F::Type{<:Union{<:UnallocatedFill,<:UnallocatedZeros}}, elt)

## With these functions defined I can print UnallocatedArrays
## compute things like sum and norm, compute the size and length
@inline Base.axes(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = axes(parent(A))
Base.size(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = size(parent(A))
FillArrays.getindex_value(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = FillArrays.getindex_value(parent(A))
Base.copy(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = A
## Can't actually use NDTensors.set_eltype because it doesn't 
## exist yet in thie area
Base.complex(A::Union{<:UnallocatedFill,<:UnallocatedZeros}) = set_alloctype(complex(parent(A)), set_parameter(alloctype(A), Position{1}(), complex(eltype(A))))
#     ## TODO Implement vec
#     # Base.vec(Z::UnallocatedZeros) = typeof(Z)(length(Z))
