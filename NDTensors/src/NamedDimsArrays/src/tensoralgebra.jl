# AbstractArray algebra
# TODO: Make this more generic based on `broadcast`.
Base.:*(a::AbstractNamedDimsArray, c::Number) = named(unname(a) * c, dimnames(a))
Base.:*(c::Number, a::AbstractNamedDimsArray) = named(c * unname(a), dimnames(a))
