# This file defines the abstract type AbstractCategory
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractCategory

using NDTensors.LabelledNumbers
using NDTensors.GradedAxes
using BlockArrays: blockedrange, blocklengths

abstract type AbstractCategory end

# ============  Base interface  =================
function Base.show(io::IO, cs::Vector{<:AbstractCategory})
  (length(cs) <= 1) && print(io, "[")
  symbol = ""
  for c in cs
    print(io, symbol, c)
    symbol = " ⊕ "
  end
  (length(cs) <= 1) && print(io, "]")
  return nothing
end

Base.isless(c1::AbstractCategory, c2::AbstractCategory) = isless(label(c1), label(c2))

# =================  Misc  ======================
function trivial(category_type::Type{<:AbstractCategory})
  return error("`trivial` not defined for type $(category_type).")
end

istrivial(c::AbstractCategory) = (c == trivial(typeof(c)))

# name conflict with LabelledNumber. TBD is that an issue?
label(c::AbstractCategory) = error("method `label` not defined for type $(typeof(c))")

# TBD dimension in Sectors or in GradedAxes namespace?
function dimension(c::AbstractCategory)
  return error("method `dimension` not defined for type $(typeof(c))")
end

function dimension(g::GradedAxes.GradedUnitRange)
  return sum(LabelledNumber.unlabel(b) * dimension(label(b)) for b in blocklengths(g))
end

function GradedAxes.dual(category_type::Type{<:AbstractCategory})
  return error("`dual` not defined for type $(category_type).")
end

# ================  fuion rule interface ====================
function label_fusion_rule(category_type::Type{<:AbstractCategory}, l1, l2)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# TBD always return GradedUnitRange?
function fusion_rule(c1::C, c2::C) where {C<:AbstractCategory}
  out = label_fusion_rule(C, label(c1), label(c2))
  if typeof(out) <: Tuple{Vector,Vector}  # TODO replace with Trait
    degen, labels = out
    # NonAbelianGroup or NonGroupCategory: return GradedUnitRange
    return GradedAxes.gradedrange(LabelledNumbers.LabelledInteger.(degen, C.(labels)))
  end
  return C(out)  # AbelianGroup: return Category
end

function fusion_rule(g::GradedAxes.GradedUnitRange, c::AbstractCategory)
  return fusion_rule(c, g)
end

function ⊗(c1::C, c2::C) where {C<:AbstractCategory}
  return fusion_rule(c1, c2)
end

# =============  fusion rule and gradedunitrange ===================
# TBD define ⊗(c, g2), ⊗(g1, c), ⊗(g1, g2)?
function GradedAxes.tensor_product(
  g1::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}}, c::C
) where {V,C<:AbstractCategory}
  g2 = gradedrange(c)
  return GradedAxes.tensor_product(g1, g2)
end

function GradedAxes.tensor_product(
  c::C, g::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}}
) where {V,C<:AbstractCategory}
  return c ⊗ g
end

function GradedAxes.tensor_product(
  g1::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}},
  g2::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}},
) where {V,C<:AbstractCategory}
  return g1 ⊗ g2
end

GradedAxes.fuse_labels(c1::AbstractCategory, c2::AbstractCategory) = c1 ⊗ c2

# ===============  sum rules ====================
⊕(c1::C, c2::C) where {C<:AbstractCategory} = GradedAxes.gradedrange([c1 => 1, c2 => 1])

function ⊕(
  c::C, g::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}}
) where {V<:Integer,C<:AbstractCategory}
  return GradedAxes.gradedrange([c => 1]) ⊕ g
end

function ⊕(
  g::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}}, c::C
) where {V<:Integer,C<:AbstractCategory}
  return g ⊕ GradedAxes.gradedrange([c => 1])
end

function ⊕(
  g1::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}},
  g2::GradedAxes.GradedUnitRange{Vector{LabelledNumbers.LabelledInteger{V,C}}},
) where {V<:Integer,C<:AbstractCategory}
  return GradedAxes.chain(g1, g2)
end
