function qr(E::Exposed)
  return qr(unexpose(E))
end
## These functions do not exist in `LinearAlgebra` but were defined
## in NDTensors. Because Expose is imported before NDTensors,
## one cannot import a these functions from NDTensors so instead
## I define them here and extend them in NDTensors
## I have done the same thing for the function cpu
## Expose.qr_positive
function qr_positive(E::Exposed)
  return qr_positive(unexpose(E))
end

## Expose.ql
function ql(E::Exposed)
  return ql(unexpose(E))
end
## Expose.ql_positive
function ql_positive(E::Exposed)
  return ql_positive(unexpose(E))
end

function LinearAlgebra.eigen(E::Exposed{<:Any,<:Union{Hermitian,Symmmetric}})
  if VERSION ≥ v"1.12-"
    # The default algorithm for eigenvalue decomposition in Julia 1.12 and later
    # changed from `syevr` (`LinearAlgebra.RobustRepresentations()`) to `syevd`
    # (`LinearAlgebra.DivideAndConquer()`) in https://github.com/JuliaLang/julia/pull/49262,
    # https://github.com/JuliaLang/julia/pull/49355.
    # However, there are cases where `syevd` fails, see:
    # https://itensor.discourse.group/t/issue-with-linearalgebra-and-julia-1-12/2362,
    # https://github.com/JuliaLang/LinearAlgebra.jl/issues/1313.
    return eigen(unexpose(E); alg=LinearAlgebra.RobustRepresentations())
  end
  return eigen(unexpose(E))
end

function LinearAlgebra.eigen(E::Exposed)
  return eigen(unexpose(E))
end

function svd(E::Exposed; kwargs...)
  return svd(unexpose(E); kwargs...)
end
