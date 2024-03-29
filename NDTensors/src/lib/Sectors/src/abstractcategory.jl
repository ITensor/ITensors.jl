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

# ================  fusion rule interface ====================
function label_fusion_rule(category_type::Type{<:AbstractCategory}, l1, l2)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# TBD always return GradedUnitRange?
function fusion_rule(c1::C, c2::C) where {C<:AbstractCategory}
  out = label_fusion_rule(C, label(c1), label(c2))
  if SymmetryStyle(c1) == AbelianGroup()
    return C(out)  # AbelianGroup: return Category
  end
  degen, labels = out
  # NonAbelianGroup or NonGroupCategory: return GradedUnitRange
  return GradedAxes.gradedrange(LabelledNumbers.LabelledInteger.(degen, C.(labels)))
end

function ⊗(c1::AbstractCategory, c2::AbstractCategory)
  return fusion_rule(c1, c2)
end

# =============  fusion rule and gradedunitrange ===================
# TBD define ⊗(c, g2), ⊗(g1, c), ⊗(g1, g2)?

# 1. make GradedAxes.tensor_product return fusion_rule
function GradedAxes.tensor_product(c1::AbstractCategory, c2::AbstractCategory)
  return fusion_rule(c1, c2)
end

function GradedAxes.tensor_product(r::AbstractUnitRange, c::AbstractCategory)
  return fusion_rule(r, c)
end

function GradedAxes.tensor_product(c::AbstractCategory, r::AbstractUnitRange)
  return fusion_rule(c, r)
end

function GradedAxes.tensor_product(
  g1::GradedAxes.GradedUnitRange, g2::GradedAxes.GradedUnitRange
)
  return fusion_rule(g1, g2)
end

# 2. make GradedAxes.fuse_labels return fusion_rule
GradedAxes.fuse_labels(c1::AbstractCategory, c2::AbstractCategory) = c1 ⊗ c2

# 3. promote Category to GradedAxes
# TODO define promote_rule
function fusion_rule(c::AbstractCategory, r::AbstractUnitRange)
  return fusion_rule(GradedAxes.gradedrange([c => 1]), r)
end

function fusion_rule(r::AbstractUnitRange, c::AbstractCategory)
  return fusion_rule(GradedAxes.gradedrange(r, [c => 1]))
end

# 4. define fusion rule for reducible representations
# TODO deal with dual
function fusion_rule(g1::GradedAxes.GradedUnitRange, g2::GradedAxes.GradedUnitRange)
  blocks3 = empty(BlockArrays.blocklengths(g1))
  for b1 in BlockArrays.blocklengths(g1)
    cat1 = LabelledNumbers.label(b1)
    degen1 = LabelledNumbers.unlabel(b1)
    for b2 in BlockArrays.blocklengths(g2)
      cat2 = LabelledNumbers.label(b2)
      degen2 = LabelledNumbers.unlabel(b2)
      degen3 = degen1 * degen2
      fuse12 = cat1 ⊗ cat2
      if SymmetryStyle(fuse12) == AbelianGroup()
        push!(blocks3, LabelledNumbers.LabelledInteger(degen3, fuse12))
      else
        g12 = BlockArrays.blocklengths(fuse12)
        # Int * LabelledInteger -> Int, need to recast explicitly
        scaled_g12 =
          LabelledNumbers.LabelledInteger.(degen3 .* g12, LabelledNumbers.label.(g12))
        append!(blocks3, scaled_g12)
      end
    end
  end
  # sort and fuse blocks carrying the same category label
  # there is probably a better way to do this
  unsorted_g3 = GradedAxes.gradedrange(blocks3)
  perm = GradedAxes.blockmergesortperm(unsorted_g3)
  vec3 = empty(blocks3)
  for b in BlockArrays.blocks(perm)
    x = unsorted_g3[b]
    n = LabelledNumbers.LabelledInteger(sum(length(x); init=0), LabelledNumbers.label(x[1]))
    push!(vec3, n)
  end
  g3 = GradedAxes.gradedrange(vec3)
  return g3
end
