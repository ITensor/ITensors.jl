using .TypeParameterAccessors: set_ndims, specify_default_parameters, unwrap_array_type

## TODO I modified this to accept any type and just match the output to the number of dims.
# for example generic_zeros(Vector, 2,3) = Matrix{Float64}[0,0,0;0,0,0;]
function generic_randn(
  arraytype::Type{<:AbstractArray}, dims...; rng=Random.default_rng()
)
  arraytype_specified = set_ndims(specify_default_parameters(unwrap_array_type(arraytype)), length(dims));
  data = similar(arraytype_specified, dims)
  return randn!(rng, data)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  arraytype_specified = set_ndims(specify_default_parameters(unwrap_array_type(arraytype)), length(dims));
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims), zero(ElT))
end
