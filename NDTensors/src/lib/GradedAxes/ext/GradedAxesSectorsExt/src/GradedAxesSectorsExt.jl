module GradedAxesSectorsExt
using ..GradedAxes: GradedAxes
using ...Sectors: Sectors, AbstractCategory, ⊗, dual

GradedAxes.fuse(c1::AbstractCategory, c2::AbstractCategory) = only(c1 ⊗ c2)

GradedAxes.dual(c::AbstractCategory) = dual(c)
end
