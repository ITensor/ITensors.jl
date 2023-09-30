SmallVectors.insert(a::Vector, index::Integer, item) = insert!(copy(a), index, item)
delete(d::AbstractDict, key) = delete!(copy(d), key)
