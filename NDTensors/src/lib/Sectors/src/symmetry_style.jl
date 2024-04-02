using BlockArrays

using NDTensors.LabelledNumbers
using NDTensors.GradedAxes

abstract type SymmetryStyle end

struct AbelianGroup <: SymmetryStyle end
struct NonAbelianGroup <: SymmetryStyle end
struct NonGroupCategory <: SymmetryStyle end
struct EmptyCategory <: SymmetryStyle end

# crash for empty g. Currently impossible to construct.
SymmetryStyle(g::AbstractUnitRange) = SymmetryStyle(LabelledNumbers.label(first(g)))
