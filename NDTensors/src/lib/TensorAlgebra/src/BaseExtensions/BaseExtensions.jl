module BaseExtensions
# `Base.indexin` doesn't handle tuples
indexin(x, y) = Base.indexin(x, y)
indexin(x, y::Tuple) = Base.indexin(x, collect(y))
end
