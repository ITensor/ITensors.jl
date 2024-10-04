#
# Trivial sector
#

using ...GradedAxes: GradedAxes

# Trivial is special as it does not have a label
struct TrivialSector <: AbstractCategory end

SymmetryStyle(::Type{TrivialSector}) = AbelianStyle()

trivial(::Type{TrivialSector}) = TrivialSector()

GradedAxes.dual(::TrivialSector) = TrivialSector()

Base.isless(::TrivialSector, ::TrivialSector) = false  # bypass default that calls label

# TrivialSector acts as trivial on any AbstractCategory
function fusion_rule(::NotAbelianStyle, ::TrivialSector, c::AbstractCategory)
  return to_gradedrange(c)
end
function fusion_rule(::NotAbelianStyle, c::AbstractCategory, ::TrivialSector)
  return to_gradedrange(c)
end

# abelian case: return Category
fusion_rule(::AbelianStyle, c::AbstractCategory, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::AbstractCategory) = c
fusion_rule(::AbelianStyle, ::TrivialSector, ::TrivialSector) = TrivialSector()

# any trivial sector equals TrivialSector
Base.:(==)(c::AbstractCategory, ::TrivialSector) = istrivial(c)
Base.:(==)(::TrivialSector, c::AbstractCategory) = istrivial(c)
Base.:(==)(::TrivialSector, ::TrivialSector) = true
