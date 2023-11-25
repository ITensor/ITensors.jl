Dictionaries.isinsertable(::AbstractSmallVector) = true
Dictionaries.isinsertable(::SmallVector) = false
Dictionaries.empty_type(::Type{SmallVector{S,T}}, ::Type{T}) where {S,T} = MSmallVector{S,T}
