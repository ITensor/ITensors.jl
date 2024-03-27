using ITensors: ITensor, apply, array, inds, itensor, permute

# Utilities for making it easier
# to define solvers (like ODE solvers)
# for TDVP

"""
    to_vec(x)
Transform `x` into a `Vector`, and return the vector, and a closure which inverts the
transformation.

Modeled after FiniteDifferences.to_vec:

https://github.com/JuliaDiff/FiniteDifferences.jl/blob/main/src/to_vec.jl
"""
to_vec(x) = error("Not implemented")

function to_vec(x::ITensor)
  function to_itensor(x_vec)
    return itensor(x_vec, inds(x))
  end
  return vec(array(x)), to_itensor
end

# Represents a time-dependent sum of terms:
#
# H(t) = f[1](t) * H0[1] + f[2](t) * H0[2] + …
#
struct TimeDependentSum{S,T}
  f::Vector{S}
  H0::T
end
TimeDependentSum(f::Vector, H0::ProjMPOSum) = TimeDependentSum(f, ITensors.terms(H0))
Base.length(H::TimeDependentSum) = length(H.f)

function Base.:*(c::Number, H::TimeDependentSum)
  return TimeDependentSum([t -> c * fₙ(t) for fₙ in H.f], H.H0)
end
Base.:*(H::TimeDependentSum, c::Number) = c * H

# Calling a `TimeDependentOpSum` at a certain time like:
#
# H(t)
#
# Returns a `ScaledSum` at that time.
(H::TimeDependentSum)(t::Number) = ScaledSum([fₙ(t) for fₙ in H.f], H.H0)

# Represents the sum of scaled terms:
#
# H = coefficient[1] * H[1] + coefficient * H[2] + …
#
struct ScaledSum{S,T}
  coefficients::Vector{S}
  H::T
end
Base.length(H::ScaledSum) = length(H.coefficients)

# Apply the scaled sum of terms:
#
# H(ψ₀) = coefficient[1] * H[1](ψ₀) + coefficient[2] * H[2](ψ₀) + …
#
# onto ψ₀.
function (H::ScaledSum)(ψ₀)
  ψ = ITensor(inds(ψ₀))
  for n in 1:length(H)
    ψ += H.coefficients[n] * apply(H.H[n], ψ₀)
  end
  return permute(ψ, inds(ψ₀))
end
