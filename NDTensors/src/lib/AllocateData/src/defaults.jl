default_initializer(::Type{<:AbstractArray}) = undef
default_eltype() = Float64
default_arraytype(elt::Type) = Array{elt}
default_arraytype() = default_arraytype(default_eltype())
