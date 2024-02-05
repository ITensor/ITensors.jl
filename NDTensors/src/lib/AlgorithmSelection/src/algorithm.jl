"""
    Algorithm

A type representing an algorithm backend for a function.

For example, a function might have multiple backend algorithm
implementations, which internally are selected with an `Algorithm` type.

This allows users to extend functionality with a new algorithm but
use the same interface.
"""
struct Algorithm{Alg,Kwargs<:NamedTuple}
  kwargs::Kwargs
end

Algorithm{Alg}(kwargs::NamedTuple) where {Alg} = Algorithm{Alg,typeof(kwargs)}(kwargs)
Algorithm{Alg}(; kwargs...) where {Alg} = Algorithm{Alg}(NamedTuple(kwargs))
Algorithm(s; kwargs...) = Algorithm{Symbol(s)}(NamedTuple(kwargs))

Algorithm(alg::Algorithm) = alg

# TODO: Use `TypeParameterAccessors`.
algorithm_string(::Algorithm{Alg}) where {Alg} = string(Alg)

function Base.show(io::IO, alg::Algorithm)
  return print(io, "Algorithm type ", algorithm_string(alg), ", ", alg.kwargs)
end
Base.print(io::IO, alg::Algorithm) = print(io, algorithm_string(alg), ", ", alg.kwargs)

"""
    @Algorithm_str

A convenience macro for writing [`Algorithm`](@ref) types, typically used when
adding methods to a function that supports multiple algorithm
backends.
"""
macro Algorithm_str(s)
  return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end
