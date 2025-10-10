using .Vendored.TypeParameterAccessors:
    unwrap_array_type,
    specify_default_type_parameters,
    specify_type_parameters,
    type_parameters

# Convert to Array, avoiding copying if possible
array(a::AbstractArray) = a
matrix(a::AbstractMatrix) = a
vector(a::AbstractVector) = a

## Warning to use these functions it is necessary to define `TypeParameterAccessors.position(::Type{<:YourArrayType}, ::typeof(ndims)))`
# Implementation, catches if `ndims(arraytype) != length(dims)`.
## TODO convert ndims to `type_parameters(::, typeof(ndims))`
function generic_randn(arraytype::Type{<:AbstractArray}, dims...; rng = Random.default_rng())
    arraytype_specified = specify_type_parameters(
        unwrap_array_type(arraytype), ndims, length(dims)
    )
    arraytype_specified = specify_default_type_parameters(arraytype_specified)
    @assert length(dims) == ndims(arraytype_specified)
    data = similar(arraytype_specified, dims...)
    return randn!(rng, data)
end

function generic_randn(
        arraytype::Type{<:AbstractArray}, dims::Tuple; rng = Random.default_rng()
    )
    return generic_randn(arraytype, dims...; rng)
end

# Implementation, catches if `ndims(arraytype) != length(dims)`.
function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
    arraytype_specified = specify_type_parameters(
        unwrap_array_type(arraytype), ndims, length(dims)
    )
    arraytype_specified = specify_default_type_parameters(arraytype_specified)
    @assert length(dims) == ndims(arraytype_specified)
    ElT = eltype(arraytype_specified)
    return fill!(similar(arraytype_specified, dims...), zero(ElT))
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims::Tuple)
    return generic_zeros(arraytype, dims...)
end
