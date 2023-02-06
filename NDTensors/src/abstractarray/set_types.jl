"""
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
function set_eltype(arraytype::Type{<:AbstractArray}, eltype::Type)
  return error(
    "Setting the element type of the array type `$arraytype` (to `$eltype`) is not currently defined.",
  )
end

"""
TODO: Use `Accessors.jl` notation:
```julia
@set ndims(arraytype) = ndims
```
"""
function set_ndims(arraytype::Type{<:AbstractArray}, ndims)
  return error(
    "Setting the number dimensions of the array type `$arraytype` (to `$ndims`) is not currently defined.",
  )
end
