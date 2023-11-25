thaw(x) = copy(x)
freeze(x) = x

thaw_type(::Type{<:AbstractArray{<:Any,N}}, ::Type{T}) where {T,N} = Array{T,N}
thaw_type(x::AbstractArray, ::Type{T}) where {T} = thaw_type(typeof(x), T)
thaw_type(x::AbstractArray{T}) where {T} = thaw_type(typeof(x), T)
