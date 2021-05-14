
# Trait to determine if an Index, Index collection, Tensor, or ITensor
# has symmetries
abstract type SymmetryStyle end

function symmetrystyle(T)
  return error("No SymmetryStyle defined for the specified object $T of type $(typeof(T))")
end

symmetrystyle(T, S, U, V...)::SymmetryStyle = (
  Base.@_inline_meta; symmetrystyle(symmetrystyle(T), symmetrystyle(S, U, V...))
)

symmetrystyle(T, S)::SymmetryStyle = symmetrystyle(symmetrystyle(T), symmetrystyle(S))

# Rules for basic collections
symmetrystyle(inds::Tuple) = symmetrystyle(inds...)
# `reduce(symmetrystyle, inds)` is not type stable for some reason
symmetrystyle(inds::AbstractVector) = mapreduce(symmetrystyle, symmetrystyle, inds)
