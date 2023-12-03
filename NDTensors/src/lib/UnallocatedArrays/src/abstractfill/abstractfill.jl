## Here are functions specifically defined for UnallocatedArrays
## not implemented by FillArrays
## TODO determine min number of functions needed to be forwarded
alloctype(A::AbstractFill) = alloctype(typeof(A))
function alloctype(Atype::Type{AbstractFill})
  return get_parameter(Atype, Position{4}())
end

allocate(A::AbstractFill) = alloctype(A)(parent(A))

## TODO Still working here I am not sure these functions and the
## Set parameter functions are working properly
function set_alloctype(F::Type{AbstractFill}, alloc::Type{<:AbstractArray})
  return set_parameter(F, Position{4}(), alloc)
end
## TODO this is broken 
function set_eltype(F::Type{AbstractFill}, elt::Type)
  return set_alloctype(set_eltype(parent(z), elt), set_eltype(alloctype(F), elt))
end

## With these functions defined I can print UnallocatedArrays
## compute things like sum and norm, compute the size and length
@inline Base.axes(A::AbstractFill) = axes(parent(A))
Base.size(A::AbstractFill) = size(parent(A))
function FillArrays.getindex_value(A::AbstractFill)
  return FillArrays.getindex_value(parent(A))
end
Base.copy(A::AbstractFill) = A
## Can't actually use NDTensors.set_eltype because it doesn't 
## exist yet in thie area
function Base.complex(A::AbstractFill)
  return set_alloctype(
    complex(parent(A)), set_parameter(alloctype(A), Position{1}(), complex(eltype(A)))
  )
end

mult_fill(a, b, val, ax) = Fill(val, ax)
mult_zeros(a, b, elt, ax) = Zeros{elt}(ax)
mult_ones(a, b, elt, ax) = Ones{elt}(ax)

broadcasted_fill(f, a, val, ax) = Fill(val, ax)
broadcasted_fill(f, a, b, val, ax) = Fill(val, ax)
broadcasted_zeros(f, a, elt, ax) = Zeros{elt}(ax)
broadcasted_zeros(f, a, b, elt, ax) = Zeros{elt}(ax)
broadcasted_ones(f, a, elt, ax) = Ones{elt}(ax)
broadcasted_ones(f, a, b, elt, ax) = Ones{elt}(ax)

kron_fill(a, b, val, ax) = Fill(val, ax)
kron_zeros(a, b, elt, ax) = Zeros{elt}(ax)
kron_ones(a, b, elt, ax) = Ones{elt}(ax)
