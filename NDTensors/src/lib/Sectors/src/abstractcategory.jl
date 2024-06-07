# This file defines the abstract type AbstractCategory
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractCategory

abstract type AbstractCategory end

# ===================================  Base interface  =====================================
function Base.isless(c1::C, c2::C) where {C<:AbstractCategory}
  return isless(category_label(c1), category_label(c2))
end

# =================================  Sectors interface  ====================================
trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
  return GradedAxes.gradedrange([trivial(eltype(axis_type))])  # always returns nondual
end
function trivial(la_type::Type{<:LabelledNumbers.LabelledInteger})
  return la_type(1, trivial(LabelledNumbers.label_type(la_type)))
end
function trivial(type::Type)
  return error("`trivial` not defined for type $(type).")
end

istrivial(c::AbstractCategory) = (c == trivial(c))

function category_label(c::AbstractCategory)
  return error("method `category_label` not defined for type $(typeof(c))")
end

block_dimensions(g::AbstractUnitRange) = block_dimensions(SymmetryStyle(g), g)
block_dimensions(::AbelianGroup, g) = GradedAxes.unlabel.(BlockArrays.blocklengths(g))
function block_dimensions(::SymmetryStyle, g)
  return Sectors.quantum_dimension.(GradedAxes.blocklabels(g)) .*
         BlockArrays.blocklengths(g)
end

quantum_dimension(x) = quantum_dimension(SymmetryStyle(x), x)

function quantum_dimension(::SymmetryStyle, c::AbstractCategory)
  return error("method `quantum_dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianGroup, ::AbstractCategory) = 1
quantum_dimension(::EmptyCategory, ::AbstractCategory) = 1
quantum_dimension(::SymmetryStyle, g::AbstractUnitRange) = sum(block_dimensions(g))
quantum_dimension(::AbelianGroup, g::AbstractUnitRange) = length(g)

# ===============================  Fusion rule interface  ==================================
⊗(c1::AbstractCategory, c2::AbstractCategory) = fusion_rule(c1, c2)

function fusion_rule(c1, c2)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::SymmetryStyle, c1::C, c2::C) where {C<:AbstractCategory}
  degen, labels = label_fusion_rule(C, category_label(c1), category_label(c2))
  return GradedAxes.gradedrange(LabelledNumbers.labelled.(degen, C.(labels)))
end

# abelian case: return Category
function fusion_rule(::AbelianGroup, c1::C, c2::C) where {C<:AbstractCategory}
  return C(label_fusion_rule(C, category_label(c1), category_label(c2)))
end

function fusion_rule(
  ::SymmetryStyle, l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger
)
  fused = LabelledNumbers.label(l1) ⊗ LabelledNumbers.label(l2)
  v =
    LabelledNumbers.labelled.(
      l1 * l2 .* BlockArrays.blocklengths(fused), GradedAxes.blocklabels(fused)
    )
  return GradedAxes.gradedrange(v)
end

function fusion_rule(
  ::AbelianGroup, l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger
)
  fused = LabelledNumbers.label(l1) ⊗ LabelledNumbers.label(l2)
  return LabelledNumbers.labelled(l1 * l2, fused)
end

function fusion_rule(
  ::EmptyCategory, l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger
)
  return LabelledNumbers.labelled(l1 * l2, sector())
end

function label_fusion_rule(category_type::Type{<:AbstractCategory}, ::Any, ::Any)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# ================================  GradedAxes interface  ==================================
# tensor_product interface
function GradedAxes.fuse_blocklengths(
  l1::LabelledNumbers.LabelledInteger{<:Integer,<:Sectors.AbstractCategory},
  l2::LabelledNumbers.LabelledInteger{<:Integer,<:Sectors.AbstractCategory},
)
  return fusion_rule(l1, l2)
end

# cast to range
to_graded_axis(c::AbstractCategory) = to_graded_axis(LabelledNumbers.labelled(1, c))
to_graded_axis(l::LabelledNumbers.LabelledInteger) = GradedAxes.gradedrange([l])
to_graded_axis(g::AbstractUnitRange) = g

# allow to fuse a category with a GradedUnitRange
function GradedAxes.tensor_product(c::AbstractCategory, g::AbstractUnitRange)
  return GradedAxes.tensor_product(to_graded_axis(c), g)
end

function GradedAxes.tensor_product(g::AbstractUnitRange, c::AbstractCategory)
  return GradedAxes.tensor_product(c, g)
end

function GradedAxes.tensor_product(c1::AbstractCategory, c2::AbstractCategory)
  return GradedAxes.tensor_product(to_graded_axis(c1), to_graded_axis(c2))
end

function GradedAxes.fusion_product(c::AbstractCategory)
  return GradedAxes.fusion_product(to_graded_axis(c))
end
