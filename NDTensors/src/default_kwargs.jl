default_maxdim(a) = minimum(size(a))
default_mindim(a) = true
default_cutoff(a) = zero(eltype(a))
default_svd_alg(a) = "divide_and_conquer"
default_use_absolute_cutoff(a) = false
default_use_relative_cutoff(a) = true
