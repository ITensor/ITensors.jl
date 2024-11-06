struct NDims{ndims} end
Base.ndims(::NDims{ndims}) where {ndims} = ndims

NDims(ndims::Integer) = NDims{ndims}()
NDims(arraytype::Type{<:AbstractArray}) = NDims(ndims(arraytype))
NDims(array::AbstractArray) = NDims(typeof(array))
