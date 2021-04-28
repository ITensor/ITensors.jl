
# Trait to determine if an Index, Index collection, Tensor, or ITensor
# has symmetries
abstract type SymmetryStyle end

symmetrystyle(T) = error("No SymmetryStyle defined for the specified type")

symmetrystyle(T, S, U, V...)::SymmetryStyle = (
  Base.@_inline_meta; symmetrystyle(symmetrystyle(T), symmetrystyle(S, U, V...))
)

symmetrystyle(T, S)::SymmetryStyle = symmetrystyle(symmetrystyle(T), symmetrystyle(S))

# Rules for basic collections
symmetrystyle(inds::Tuple) = symmetrystyle(inds...)
symmetrystyle(inds::AbstractVector)::SymmetryStyle = symmetrystyle(inds...)
