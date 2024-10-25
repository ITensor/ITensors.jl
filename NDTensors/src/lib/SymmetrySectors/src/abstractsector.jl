# This file defines the abstract type AbstractSector
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractSector

using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, label_type, labelled, unlabel, unlabel_type
using ..GradedAxes: GradedAxes, blocklabels, fuse_blocklengths, gradedrange, tensor_product

abstract type AbstractSector end

# ===================================  Base interface  =====================================
function Base.isless(c1::C, c2::C) where {C<:AbstractSector}
  return isless(sector_label(c1), sector_label(c2))
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

istrivial(c::AbstractSector) = (c == trivial(c))

function sector_label(c::AbstractSector)
  return error("method `sector_label` not defined for type $(typeof(c))")
end

block_dimensions(g::AbstractUnitRange) = block_dimensions(SymmetryStyle(g), g)
block_dimensions(::AbelianStyle, g) = unlabel.(blocklengths(g))
function block_dimensions(::NotAbelianStyle, g)
  return quantum_dimension.(blocklabels(g)) .* blocklengths(g)
end

quantum_dimension(x) = quantum_dimension(SymmetryStyle(x), x)

function quantum_dimension(::NotAbelianStyle, c::AbstractSector)
  return error("method `quantum_dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianStyle, ::AbstractSector) = 1
quantum_dimension(::AbelianStyle, g::AbstractUnitRange) = length(g)
quantum_dimension(::NotAbelianStyle, g::AbstractUnitRange) = sum(block_dimensions(g))

# ===============================  Fusion rule interface  ==================================
⊗(c1::AbstractSector, c2::AbstractSector) = fusion_rule(c1, c2)

function fusion_rule(c1::AbstractSector, c2::AbstractSector)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::NotAbelianStyle, c1::C, c2::C) where {C<:AbstractSector}
  sector_degen_pairs = label_fusion_rule(C, sector_label(c1), sector_label(c2))
  return gradedrange(sector_degen_pairs)
end

# abelian case: return Sector
function fusion_rule(::AbelianStyle, c1::C, c2::C) where {C<:AbstractSector}
  return label(only(fusion_rule(NotAbelianStyle(), c1, c2)))
end

function label_fusion_rule(sector_type::Type{<:AbstractSector}, l1, l2)
  return [abelian_label_fusion_rule(sector_type, l1, l2) => 1]
end

# ================================  GradedAxes interface  ==================================
# tensor_product interface
function GradedAxes.fuse_blocklengths(
  l1::LabelledInteger{<:Integer,<:AbstractSector},
  l2::LabelledInteger{<:Integer,<:AbstractSector},
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
to_gradedrange(c::AbstractSector) = to_gradedrange(labelled(1, c))
to_gradedrange(l::LabelledInteger) = gradedrange([l])
to_gradedrange(g::AbstractUnitRange) = g

# allow to fuse a Sector with a GradedUnitRange
function GradedAxes.tensor_product(c::AbstractSector, g::AbstractUnitRange)
  return tensor_product(to_gradedrange(c), g)
end

function GradedAxes.tensor_product(g::AbstractUnitRange, c::AbstractSector)
  return tensor_product(g, to_gradedrange(c))
end

function GradedAxes.tensor_product(c1::AbstractSector, c2::AbstractSector)
  return to_gradedrange(fusion_rule(c1, c2))
end

function GradedAxes.fusion_product(c::AbstractSector)
  return to_gradedrange(c)
end
