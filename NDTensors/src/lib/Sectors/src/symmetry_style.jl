using BlockArrays

using NDTensors.LabelledNumbers
using NDTensors.GradedAxes

abstract type SymmetryStyle end

struct AbelianGroup <: SymmetryStyle end
struct NonAbelianGroup <: SymmetryStyle end
struct NonGroupCategory <: SymmetryStyle end
struct EmptyCategory <: SymmetryStyle end

combine_styles(::AbelianGroup, ::AbelianGroup) = AbelianGroup()
combine_styles(::AbelianGroup, ::NonAbelianGroup) = NonAbelianGroup()
combine_styles(::AbelianGroup, ::NonGroupCategory) = NonGroupCategory()
combine_styles(::NonAbelianGroup, ::AbelianGroup) = NonAbelianGroup()
combine_styles(::NonAbelianGroup, ::NonAbelianGroup) = NonAbelianGroup()
combine_styles(::NonAbelianGroup, ::NonGroupCategory) = NonGroupCategory()
combine_styles(::NonGroupCategory, ::SymmetryStyle) = NonGroupCategory()
combine_styles(::EmptyCategory, s::SymmetryStyle) = s
combine_styles(s::SymmetryStyle, ::EmptyCategory) = s
combine_styles(::EmptyCategory, ::EmptyCategory) = EmptyCategory()

SymmetryStyle(l::LabelledNumbers.LabelledInteger) = SymmetryStyle(LabelledNumbers.label(l))

# crash for empty g. Currently impossible to construct.
SymmetryStyle(g::AbstractUnitRange) = SymmetryStyle(first(g))
