using ..Vendored.TypeParameterAccessors: unwrap_array_type
replace_nothing(::Nothing, replacement) = replacement
replace_nothing(value, replacement) = value

default_maxdim(a) = minimum(size(a))
default_mindim(a) = true
default_cutoff(a) = zero(eltype(a))
default_svd_alg(a) = default_svd_alg(unwrap_array_type(a), a)
default_svd_alg(::Type{<:AbstractArray}, a) = "divide_and_conquer"
default_use_absolute_cutoff(a) = false
default_use_relative_cutoff(a) = true
