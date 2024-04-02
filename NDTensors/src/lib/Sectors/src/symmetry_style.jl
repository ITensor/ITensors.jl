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

quantum_dimension(c) = quantum_dimension(SymmetryStyle(c), c)

quantum_dimension(::AbelianGroup, g::AbstractUnitRange) = length(g)

function quantum_dimension(::NonAbelianGroup, g::GradedAxes.GradedUnitRange)
  return sum(
    LabelledNumbers.unlabel(b) * quantum_dimension(LabelledNumbers.label(b)) for
    b in BlockArrays.blocklengths(g), init in 0
  )
end

function quantum_dimension(::NonGroupCategory, g::GradedAxes.GradedUnitRange)
  return sum(
    LabelledNumbers.unlabel(b) * quantum_dimension(LabelledNumbers.label(b)) for
    b in BlockArrays.blocklengths(g), init in 0.0
  )
end
