using .Vendored.TypeParameterAccessors: TypeParameterAccessors

"""
# Do we still want to define things like this?
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
# This is for uniform `Diag` storage which uses
# a Number as the data type.
# TODO: Delete this when we change to using a
# `FillArray` instead. This is a stand-in
# to make things work with the current design.
function TypeParameterAccessors.set_ndims(numbertype::Type{<:Number}, ndims)
    return numbertype
end
