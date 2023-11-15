# TODO: This needs to be done more carefully by
# grabbing upper or lower triangles of the parent matrix.
nonzero_keys(a::Hermitian) = nonzero_keys(parent(a))
