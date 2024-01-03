conj(as::AliasStyle, A::AbstractArray) = conj(A)
conj(as::AllowAlias, A::Array{<:Real}) = A
