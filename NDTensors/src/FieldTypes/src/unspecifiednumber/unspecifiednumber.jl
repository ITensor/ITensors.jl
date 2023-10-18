abstract type AbstractUnspecifiedNumber <: Number end

struct UnspecifiedNumber{T} <: AbstractUnspecifiedNumber
  value::T
end
