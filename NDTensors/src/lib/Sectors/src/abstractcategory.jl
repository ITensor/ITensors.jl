# This file defines the abstract type AbstractCategory
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractCategory

using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, label_type, labelled, unlabel, unlabel_type
using ..GradedAxes: GradedAxes, blocklabels, fuse_blocklengths, gradedrange, tensor_product

abstract type AbstractCategory end

# ===================================  Base interface  =====================================
function Base.isless(c1::C, c2::C) where {C<:AbstractCategory}
  return isless(category_label(c1), category_label(c2))
end

# =================================  Sectors interface  ====================================
trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
  return gradedrange([trivial(eltype(axis_type))])  # always returns nondual
end
function trivial(la_type::Type{<:LabelledInteger})
  return labelled(one(unlabel_type(la_type)), trivial(label_type(la_type)))
end
function trivial(type::Type)
  return error("`trivial` not defined for type $(type).")
end

istrivial(c::AbstractCategory) = (c == trivial(c))

function category_label(c::AbstractCategory)
  return error("method `category_label` not defined for type $(typeof(c))")
end

block_dimensions(g::AbstractUnitRange) = block_dimensions(SymmetryStyle(g), g)
block_dimensions(::AbelianStyle, g) = unlabel.(blocklengths(g))
function block_dimensions(::NotAbelianStyle, g)
  return quantum_dimension.(blocklabels(g)) .* blocklengths(g)
end

quantum_dimension(x) = quantum_dimension(SymmetryStyle(x), x)

function quantum_dimension(::NotAbelianStyle, c::AbstractCategory)
  return error("method `quantum_dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianStyle, ::AbstractCategory) = 1
quantum_dimension(::AbelianStyle, g::AbstractUnitRange) = length(g)
quantum_dimension(::NotAbelianStyle, g::AbstractUnitRange) = sum(block_dimensions(g))

# ===============================  Fusion rule interface  ==================================
⊗(c1::AbstractCategory, c2::AbstractCategory) = fusion_rule(c1, c2)

function fusion_rule(c1::AbstractCategory, c2::AbstractCategory)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::NotAbelianStyle, c1::C, c2::C) where {C<:AbstractCategory}
  degen, sectors = label_fusion_rule(C, category_label(c1), category_label(c2))
  return gradedrange(labelled.(degen, sectors))
end

# abelian case: return Category
function fusion_rule(::AbelianStyle, c1::C, c2::C) where {C<:AbstractCategory}
  _, sectors = label_fusion_rule(C, category_label(c1), category_label(c2))
  return only(sectors)
end

function label_fusion_rule(category_type::Type{<:AbstractCategory}, ::Any, ::Any)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# ================================  GradedAxes interface  ==================================
# tensor_product interface
function GradedAxes.fuse_blocklengths(
  l1::LabelledInteger{<:Integer,<:AbstractCategory},
  l2::LabelledInteger{<:Integer,<:AbstractCategory},
)
  return fuse_blocklengths(combine_styles(SymmetryStyle(l1), SymmetryStyle(l2)), l1, l2)
end

function GradedAxes.fuse_blocklengths(
  ::NotAbelianStyle, l1::LabelledInteger, l2::LabelledInteger
)
  fused = label(l1) ⊗ label(l2)
  v = labelled.(l1 * l2 .* blocklengths(fused), blocklabels(fused))
  return gradedrange(v)
end

function GradedAxes.fuse_blocklengths(
  ::AbelianStyle, l1::LabelledInteger, l2::LabelledInteger
)
  fused = label(l1) ⊗ label(l2)
  return gradedrange([labelled(l1 * l2, fused)])
end

# cast to range
to_gradedrange(c::AbstractCategory) = to_gradedrange(labelled(1, c))
to_gradedrange(l::LabelledInteger) = gradedrange([l])
to_gradedrange(g::AbstractUnitRange) = g

# allow to fuse a category with a GradedUnitRange
function GradedAxes.tensor_product(c::AbstractCategory, g::AbstractUnitRange)
  return tensor_product(to_gradedrange(c), g)
end

function GradedAxes.tensor_product(g::AbstractUnitRange, c::AbstractCategory)
  return tensor_product(g, to_gradedrange(c))
end

function GradedAxes.tensor_product(c1::AbstractCategory, c2::AbstractCategory)
  return to_gradedrange(fusion_rule(c1, c2))
end

function GradedAxes.fusion_product(c::AbstractCategory)
  return to_gradedrange(c)
end
