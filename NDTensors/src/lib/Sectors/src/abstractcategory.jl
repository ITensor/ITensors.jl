# This file defines the abstract type AbstractCategory
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractCategory

abstract type AbstractCategory end

# ============  Base interface  =================
Base.isless(c1::AbstractCategory, c2::AbstractCategory) = isless(label(c1), label(c2))

# =================  Misc  ======================
function trivial(category_type::Type{<:AbstractCategory})
  return error("`trivial` not defined for type $(category_type).")
end

istrivial(c::AbstractCategory) = (c == trivial(typeof(c)))

# name conflict with LabelledNumber.label. TBD is that an issue?
label(c::AbstractCategory) = error("method `label` not defined for type $(typeof(c))")

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
  mult = LabelledNumbers.unlabel.(BlockArrays.blocklengths(g))
  dims = quantum_dimension.(LabelledNumbers.label.(BlockArrays.blocklengths(g)))
  return sum(m * d for (m, d) in zip(mult, dims))
end

quantum_dimension(::AbelianGroup, g::AbstractUnitRange) = length(g)

# ================  fusion rule interface ====================
⊗(c1::AbstractCategory, c2::AbstractCategory) = fusion_rule(c1, c2)

function fusion_rule(c1::C, c2::C) where {C<:AbstractCategory}
  return fusion_rule(SymmetryStyle(c1), c1, c2)
end

function fusion_rule(::SymmetryStyle, c1::C, c2::C) where {C<:AbstractCategory}
  degen, labels = label_fusion_rule(C, label(c1), label(c2))
  return GradedAxes.gradedrange(LabelledNumbers.LabelledInteger.(degen, C.(labels)))
end

function fusion_rule(::AbelianGroup, c1::C, c2::C) where {C<:AbstractCategory}
  return C(label_fusion_rule(C, label(c1), label(c2)))  # return AbelianGroup
end

function label_fusion_rule(category_type::Type{<:AbstractCategory}, l1, l2)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# =============  fusion rule and gradedunitrange ===================
to_graded_axis(c::AbstractCategory) = GradedAxes.gradedrange([c => 1])
to_graded_axis(g::AbstractUnitRange) = g

function GradedAxes.fusion_product(a, b)
  return GradedAxes.fusion_product(to_graded_axis(a), to_graded_axis(b))
end

# TODO deal with dual
function GradedAxes.fusion_product(g1::AbstractUnitRange, g2::AbstractUnitRange)
  blocks2 = BlockArrays.blocklengths(g2)
  blocks3 = empty(blocks2)
  sym = SymmetryStyle(g2)
  for b2 in blocks2
    c2 = LabelledNumbers.label(b2)
    degen2 = LabelledNumbers.unlabel(b2)
    for b1 in BlockArrays.blocklengths(g1)
      c1 = LabelledNumbers.label(b1)
      degen1 = LabelledNumbers.unlabel(b1)
      degen3 = degen1 * degen2
      _append_fusion!(blocks3, sym, degen3, c1, c2)
    end
  end
  la3 = LabelledNumbers.label.(blocks3)
  pairs3 = [r => sum(blocks3[findall(==(r), la3)]; init=0) for r in sort(unique(la3))]
  return GradedAxes.gradedrange(pairs3)
end

function _append_fusion!(blocks3, ::AbelianGroup, degen3, c1::C, c2::C) where {C}
  return push!(blocks3, LabelledNumbers.LabelledInteger(degen3, fusion_rule(c1, c2)))
end
function _append_fusion!(blocks3, ::SymmetryStyle, degen3, c1::C, c2::C) where {C}
  fused_blocks = BlockArrays.blocklengths(fusion_rule(c1, c2))
  g12 =
    LabelledNumbers.LabelledInteger.(
      degen3 * LabelledNumbers.unlabel.(fused_blocks), LabelledNumbers.label.(fused_blocks)
    )
  return append!(blocks3, g12)
end
