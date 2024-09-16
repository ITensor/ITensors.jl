using BlockArrays: AbstractBlockedUnitRange
using ..LabelledNumbers: islabelled

# Represents the range `1:1` or `Base.OneTo(1)`.
struct OneToOne{T} <: AbstractUnitRange{T} end
OneToOne() = OneToOne{Bool}()
Base.first(a::OneToOne) = one(eltype(a))
Base.last(a::OneToOne) = one(eltype(a))

# == is just a range comparison that ignores labels. Need dedicated function to check equality.
gradedisequal(::AbstractBlockedUnitRange, ::AbstractUnitRange) = false
gradedisequal(::AbstractUnitRange, ::AbstractBlockedUnitRange) = false
gradedisequal(::AbstractBlockedUnitRange, ::OneToOne) = false
gradedisequal(::OneToOne, ::AbstractBlockedUnitRange) = false
function gradedisequal(a1::AbstractBlockedUnitRange, a2::AbstractBlockedUnitRange)
  return blockisequal(a1, a2)
end
function gradedisequal(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  return blockisequal(a1, a2) && (blocklabels(a1) == blocklabels(a2))
end
gradedisequal(::GradedUnitRangeDual, ::GradedUnitRange) = false
gradedisequal(::GradedUnitRange, ::GradedUnitRangeDual) = false
function gradedisequal(a1::GradedUnitRangeDual, a2::GradedUnitRangeDual)
  return gradedisequal(nondual(a1), nondual(a2))
end

gradedisequal(::OneToOne, ::OneToOne) = true

function gradedisequal(::OneToOne, g::AbstractUnitRange)
  return !islabelled(eltype(g)) && (first(g) == last(g) == 1)
end
gradedisequal(g::AbstractUnitRange, a0::OneToOne) = gradedisequal(a0, g)

gradedisequal(::UnitRangeDual, ::AbstractUnitRange) = false
gradedisequal(::AbstractUnitRange, ::UnitRangeDual) = false
gradedisequal(::OneToOne, ::UnitRangeDual) = false
gradedisequal(::UnitRangeDual, ::OneToOne) = false
function gradedisequal(a1::UnitRangeDual, a2::UnitRangeDual)
  return gradedisequal(nondual(a1), nondual(a2))
end

gradedisequal(a1::AbstractUnitRange, a2::AbstractUnitRange) = a1 == a2
