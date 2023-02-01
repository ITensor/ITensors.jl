"""
    Algorithm

A type representing an algorithm backend for a function.

For example, ITensor provides multiple backend algorithms for contracting
an MPO with an MPS, which internally are selected with an `Algorithm` type.

This allows users to extend functions in ITensor with new algorithms, but
use the same interface.
"""
struct Algorithm{Alg} end

Algorithm(s) = Algorithm{Symbol(s)}()
algorithm_string(::Algorithm{Alg}) where {Alg} = string(Alg)

show(io::IO, alg::Algorithm) = print(io, "Algorithm type ", algorithm_string(alg))
print(io::IO, ::Algorithm{Alg}) where {Alg} = print(io, Alg)

"""
    @Algorithm_str

A convenience macro for writing [`Algorithm`](@ref) types, typically used when
adding methods to a function in ITensor that supports multiple algorithm
backends (like contracting an MPO with an MPS).
"""
macro Algorithm_str(s)
  return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end
