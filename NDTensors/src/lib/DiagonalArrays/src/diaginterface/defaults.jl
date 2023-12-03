using Compat: Returns

default_size(diag::AbstractVector, n) = ntuple(Returns(length(diag)), n)
