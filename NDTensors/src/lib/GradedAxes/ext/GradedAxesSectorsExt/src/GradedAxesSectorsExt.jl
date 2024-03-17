module GradedAxesSectorsExt
using ..GradedAxes: GradedAxes
using ...Sectors: Sectors, AbstractCategory, ⊗ # , dual

GradedAxes.fuse_labels(c1::AbstractCategory, c2::AbstractCategory) = only(c1 ⊗ c2)

# TODO: Decide the fate of `dual`.
## GradedAxes.dual(c::AbstractCategory) = dual(c)
end
