"""
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
function set_eltype(arraytype::Type{<:Array}, eltype::Type)
  return Array{eltype,ndims(arraytype)}
end

"""
TODO: Use `Accessors.jl` notation:
```julia
@set ndims(arraytype) = ndims
```
"""
function set_ndims(arraytype::Type{<:Array}, ndims)
  return Array{eltype(arraytype),ndims}
end
