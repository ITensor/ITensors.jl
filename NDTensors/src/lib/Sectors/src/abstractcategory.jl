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
  return GradedAxes.gradedrange(LabelledNumbers.LabelledInteger.(degen, C.(labels)))
end

# abelian case: return Category
function fusion_rule(::AbelianGroup, c1::C, c2::C) where {C<:AbstractCategory}
  return C(label_fusion_rule(C, category_label(c1), category_label(c2)))
end

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

function fusion_rule(
  ::EmptyCategory, l1::LabelledNumbers.LabelledInteger, l2::LabelledNumbers.LabelledInteger
)
  return LabelledNumbers.LabelledInteger(l1 * l2, sector())
end

function label_fusion_rule(category_type::Type{<:AbstractCategory}, ::Any, ::Any)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

# ================================  GradedAxes interface  ==================================
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
function GradedAxes.fuse_labels(::EmptyCategory, c1::AbstractCategory, c2::AbstractCategory)
  return fusion_rule(c1, c2)
end

# cast to range
to_graded_axis(c::AbstractCategory) = to_graded_axis(LabelledNumbers.LabelledInteger(1, c))
to_graded_axis(l::LabelledNumbers.LabelledInteger) = GradedAxes.gradedrange([l])
to_graded_axis(g::AbstractUnitRange) = g

# allow to fuse a category with a GradedUnitRange
function GradedAxes.fusion_product(a, b)
  return GradedAxes.fusion_product(to_graded_axis(a), to_graded_axis(b))
end

# fusion_product with one input to be used in generic fusion_product(Tuple...)
# TBD define fusion_product() = gradedrange([sector(())=>1])?
GradedAxes.fusion_product(x) = GradedAxes.fusion_product(to_graded_axis(x))

# product with trivial = easy handling of UnitRangeDual + sort and merge blocks
GradedAxes.fusion_product(g::AbstractUnitRange) = GradedAxes.fusion_product(trivial(g), g)

function GradedAxes.fusion_product(
  g1::GradedAxes.GradedUnitRange, g2::GradedAxes.GradedUnitRange
)
  nested_blocks = map(
    ((l1, l2),) -> to_graded_axis(fusion_rule(l1, l2)),
    Iterators.product(BlockArrays.blocklengths(g1), BlockArrays.blocklengths(g2)),
  )
  blocks12 = reduce(vcat, BlockArrays.blocklengths.(nested_blocks))
  la3 = LabelledNumbers.label.(blocks12)
  pairs3 = [r => sum(blocks12[findall(==(r), la3)]; init=0) for r in sort(unique(la3))]
  out = GradedAxes.gradedrange(pairs3)
  return out
end
