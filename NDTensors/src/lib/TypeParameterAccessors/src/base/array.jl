position(::Type{<:Array}, ::typeof(eltype)) = Position(1)
position(::Type{<:Array}, ::typeof(ndims)) = Position(2)
