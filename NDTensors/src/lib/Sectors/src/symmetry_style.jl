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

quantum_dimension(c::Any) = quantum_dimension(SymmetryStyle(c), c)

function quantum_dimension(::SymmetryStyle, ::Any)
  return error("method `dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianGroup, ::Any) = 1
quantum_dimension(::EmptyCategory, ::Any) = 0

function quantum_dimension(g::AbstractUnitRange)
  if SymmetryStyle(g) == AbelianGroup()
    return length(g)
  elseif SymmetryStyle(g) == NonAbelianGroup()
    return sum(
      LabelledNumbers.unlabel(b) * quantum_dimension(LabelledNumbers.label(b)) for
      b in BlockArrays.blocklengths(g), init in 0
    )
  else
    return sum(
      LabelledNumbers.unlabel(b) * quantum_dimension(LabelledNumbers.label(b)) for
      b in BlockArrays.blocklengths(g), init in 0.0
    )
  end
end
