"""
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
function set_eltype(arraytype::Type{<:Array}, eltype::Type)
  return set_parameters(arraytype, Position(1), eltype)
end

"""
TODO: Use `Accessors.jl` notation:
```julia
@set ndims(arraytype) = ndims
```
"""
function set_ndims(arraytype::Type{<:Array}, ndims)
  return set_parameters(arraytype, Position(2), ndims)
end

# SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
function set_eltype(arraytype::Type{<:SubArray}, eltype::Type)
  arraytype_1 = set_parameters(arraytype, Position(1), eltype)
  parent_arraytype = get_parameter(arraytype, Position(3))
  parent_arraytype_1 = set_eltype(parent_arraytype, eltype)
  return set_parameters(arraytype_1, Position(3), parent_arraytype_1)
end

# TODO: Figure out how to define this properly.
# function set_ndims(arraytype::Type{<:SubArray}, ndims)
#   arraytype_1 = set_parameters(arraytype, Position(2), ndims)
#   parent_arraytype = get_parameter(arraytype, Position(3))
#   parent_arraytype_1 = set_ndims(parent_arraytype, ndims)
#   return set_parameters(arraytype_1, Position(3), parent_arraytype_1)
# end
