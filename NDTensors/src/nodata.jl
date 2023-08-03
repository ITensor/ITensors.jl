# Denotes when a storage type has no data
struct NoData end

size(::NoData) = (0,)
length(::NoData) = 0
fill!(::NoData, ::EmptyNumber) = NoData()
