# Represents a zero value and an index
# TODO: Rename `GetIndexZero`?
struct Zero end
(::Zero)(type::Type, I) = zero(type)
