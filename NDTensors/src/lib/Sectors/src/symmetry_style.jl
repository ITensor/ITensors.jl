# This file defines SymmetryStyle, a trait to distinguish abelian groups, non-abelian groups
# and non-group fusion categories.

using ..LabelledNumbers: LabelledInteger, label

abstract type SymmetryStyle end

struct AbelianStyle <: SymmetryStyle end
struct NotAbelianStyle <: SymmetryStyle end

combine_styles(::AbelianStyle, ::AbelianStyle) = AbelianStyle()
combine_styles(::SymmetryStyle, ::SymmetryStyle) = NotAbelianStyle()

SymmetryStyle(l::LabelledInteger) = SymmetryStyle(label(l))

# crash for empty g. Currently impossible to construct.
SymmetryStyle(g::AbstractUnitRange) = SymmetryStyle(first(g))
