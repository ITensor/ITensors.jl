# Represents a zero value and an index
# TODO: Rename `GetIndexZero`?
struct Zero end
(f::Zero)(a::AbstractArray, I) = f(eltype(a), I)
(::Zero)(type::Type, I) = zero(type)
