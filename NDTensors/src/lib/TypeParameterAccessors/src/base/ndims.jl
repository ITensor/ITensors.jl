## NDTensors.ndims (not imported from Base)

## TODO So here I try to use the new type_parameters system for `NDTensors.ndims`
## But if `ndims` is not defined for a type, I revert to using Base.ndims
function TypeParameterAccessors.ndims(array)
    type_parameter(array, Base.ndims)
end

# ## In house patch to deal issue of calling ndims with an Array of unspecified eltype
# ## https://github.com/JuliaLang/julia/pull/40682
# if VERSION < v"1.7"
#   TypeParameterAccessors.ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = N
# end
