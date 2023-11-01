function qr(E::Exposed)
  return qr(unexpose(E))
end
## These functions do not exist in `LinearAlgebra` but were defined
## in NDTensors. Because Unwrap is imported before NDTensors,
## one cannot import a these functions from NDTensors so instead
## I define them here and extend them in NDTensors
## I have done the same thing for the function cpu
## Unwrap.qr_positive
function qr_positive(E::Exposed)
  return qr_positive(unexpose(E))
end

## Unwrap.ql
function ql(E::Exposed)
  return ql(unexpose(E))
end
## Unwrap.ql_positive
function ql_positive(E::Exposed)
  return ql_positive(unexpose(E))
end

function eigen(E::Exposed)
  return eigen(unexpose(E))
end

function svd(E::Exposed; kwargs...)
  return svd(unexpose(E); kwargs...)
end
