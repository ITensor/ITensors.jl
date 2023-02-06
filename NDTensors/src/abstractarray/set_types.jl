"""
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
set_eltype(arraytype::Type{<:AbstractArray}, eltype::Type) = error("Setting the element type of the array type `$arraytype` (to `$eltype`) is not currently defined.")

"""
TODO: Use `Accessors.jl` notation:
```julia
@set ndims(arraytype) = ndims
```
"""
set_ndims(arraytype::Type{<:AbstractArray}, ndims) = error("Setting the number dimensions of the array type `$arraytype` (to `$ndims`) is not currently defined.")
