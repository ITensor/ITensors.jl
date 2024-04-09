# This file defines the abstract type AbstractCategory
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractCategory

abstract type AbstractCategory end

# ============  Base interface  =================
function Base.isless(c1::C, c2::C) where {C<:AbstractCategory}
  return isless(category_label(c1), category_label(c2))
end

# =================  Misc  ======================
function trivial(category_type::Type{<:AbstractCategory})
  return error("`trivial` not defined for type $(category_type).")
end

istrivial(c::AbstractCategory) = (c == trivial(typeof(c)))

function category_label(c::AbstractCategory)
  return error("method `category_label` not defined for type $(typeof(c))")
end

function GradedAxes.dual(category_type::Type{<:AbstractCategory})
  return error("`dual` not defined for type $(category_type).")
end

quantum_dimension(x) = quantum_dimension(SymmetryStyle(x), x)

function quantum_dimension(::SymmetryStyle, c::AbstractCategory)
  return error("method `quantum_dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianGroup, ::AbstractCategory) = 1
quantum_dimension(::EmptyCategory, ::AbstractCategory) = 0

function quantum_dimension(::SymmetryStyle, g::AbstractUnitRange)
  gblocks = BlockArrays.blocklengths(g)
  return sum(gblocks .* quantum_dimension.(LabelledNumbers.label.(gblocks)))
end

quantum_dimension(::AbelianGroup, g::AbstractUnitRange) = length(g)
function quantum_dimension(::SymmetryStyle, g::GradedAxes.UnitRangeDual)
  return quantum_dimension(GradedAxes.dual(g))
end
quantum_dimension(::AbelianGroup, g::GradedAxes.UnitRangeDual) = length(g)  # resolves ambiguity

# ================  fusion rule interface ====================
⊗(c1::AbstractCategory, c2::AbstractCategory) = fusion_rule(c1, c2)

function fusion_rule(c1, c2)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::SymmetryStyle, c1::C, c2::C) where {C<:AbstractCategory}
  degen, labels = label_fusion_rule(C, category_label(c1), category_label(c2))
  return GradedAxes.gradedrange(LabelledNumbers.LabelledInteger.(degen, C.(labels)))
end

# abelian case: return Category
function fusion_rule(::AbelianGroup, c1::C, c2::C) where {C<:AbstractCategory}
  return C(label_fusion_rule(C, category_label(c1), category_label(c2)))
end

function label_fusion_rule(category_type::Type{<:AbstractCategory}, l1, l2)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# convenient to define fusion rule for LabelledInteger too
# TBD expose this through ⊗? Currently not accessible.
function fusion_rule(
  ::SymmetryStyle, l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger
)
  blocks12 = BlockArrays.blocklengths(LabelledNumbers.label(l1) ⊗ LabelledNumbers.label(l2))
  v =
    LabelledNumbers.LabelledInteger.(l1 * l2 .* blocks12, LabelledNumbers.label.(blocks12))
  return GradedAxes.gradedrange(v)
end

function fusion_rule(
  ::AbelianGroup, l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger
)
  fused = LabelledNumbers.label(l1) ⊗ LabelledNumbers.label(l2)
  return LabelledNumbers.LabelledInteger(l1 * l2, fused)
end

# =============  fusion rule and gradedunitrange ===================
# GradedAxes.tensor_product interface. Only for abelian groups.
function GradedAxes.fuse_labels(c1::AbstractCategory, c2::AbstractCategory)
  return GradedAxes.fuse_labels(
    combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2
  )
end
function GradedAxes.fuse_labels(::SymmetryStyle, c1::AbstractCategory, c2::AbstractCategory)
  return error("`fuse_labels` is only defined for abelian groups")
end
function GradedAxes.fuse_labels(::AbelianGroup, c1::AbstractCategory, c2::AbstractCategory)
  return fusion_rule(c1, c2)
end

# cast to range
to_graded_axis(c::AbstractCategory) = GradedAxes.gradedrange([c => 1])
to_graded_axis(l::LabelledNumbers.LabelledInteger) = GradedAxes.gradedrange([l])
to_graded_axis(g::AbstractUnitRange) = g

# allow to fuse a category with a GradedUnitRange
function GradedAxes.fusion_product(a, b)
  return GradedAxes.fusion_product(to_graded_axis(a), to_graded_axis(b))
end

function GradedAxes.fusion_product(
  g1::BlockArrays.BlockedUnitRange, g2::BlockArrays.BlockedUnitRange
)
  blocks12 = Vector{eltype(to_graded_axis(fusion_rule(first(g1), first(g2))))}()
  for l1 in BlockArrays.blocklengths(g1)
    for l2 in BlockArrays.blocklengths(g2)
      append!(blocks12, BlockArrays.blocklengths(to_graded_axis(fusion_rule(l1, l2))))
    end
  end
  la3 = LabelledNumbers.label.(blocks12)
  pairs3 = [r => sum(blocks12[findall(==(r), la3)]; init=0) for r in sort(unique(la3))]
  out = GradedAxes.gradedrange(pairs3)
  return out
end
