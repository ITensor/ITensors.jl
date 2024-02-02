using .BaseExtensions: _permutedims, _permutedims!

abstract type FusionStyle end

struct ReshapeFusion <: FusionStyle end
struct BlockReshapeFusion <: FusionStyle end
struct SectorFusion <: FusionStyle end

# Defaults to a simple reshape
combine_fusion_styles(style1::Style, style2::Style) where {Style<:FusionStyle} = Style()
combine_fusion_styles(style1::FusionStyle, style2::FusionStyle) = ReshapeFusion()
combine_fusion_styles(styles::FusionStyle...) = foldl(combine_fusion_styles, styles)
FusionStyle(axis::AbstractUnitRange) = ReshapeFusion()
function FusionStyle(axes::Tuple{Vararg{<:AbstractUnitRange}})
  return combine_fusion_styles(FusionStyle.(axes)...)
end
FusionStyle(a::AbstractArray) = FusionStyle(axes(a))

# Overload this version for most arrays
function fusedims(::ReshapeFusion, a::AbstractArray, axes::AbstractUnitRange...)
  return reshape(a, axes)
end

⊗(a::AbstractUnitRange) = a
function ⊗(a1::AbstractUnitRange, a2::AbstractUnitRange, as::AbstractUnitRange...)
  return ⊗(a1, ⊗(a2, as...))
end
⊗(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) * length(a2))
⊗() = Base.OneTo(1)

# Overload this version for most arrays
function fusedims(a::AbstractArray, axes::AbstractUnitRange...)
  return fusedims(FusionStyle(a), a, axes...)
end

# Overload this version for fusion tensors, array maps, etc.
function fusedims(a::AbstractArray, axesblocks::Tuple{Vararg{AbstractUnitRange}}...)
  return fusedims(a, flatten_tuples(axesblocks)...)
end

# Fix ambiguity issue
fusedims(a::AbstractArray{<:Any,0}, ::Vararg{Tuple{}}) = a

# TODO: Is this needed? Maybe delete.
function fusedims(a::AbstractArray, permblocks...)
  return fusedims(a, blockedperm(permblocks...; length=Val(ndims(a))))
end

function fuseaxes(
  axes::Tuple{Vararg{AbstractUnitRange}}, blockedperm::AbstractBlockedPermutation
)
  axesblocks = blockpermute(axes, blockedperm)
  return map(block -> ⊗(block...), axesblocks)
end

function fuseaxes(a::AbstractArray, blockedperm::AbstractBlockedPermutation)
  return fuseaxes(axes(a), blockedperm)
end

# Fuse adjacent dimensions
function fusedims(a::AbstractArray, blockedperm::BlockedTrivialPermutation)
  axes_fused = fuseaxes(a, blockedperm)
  return fusedims(a, axes_fused)
end

function fusedims(a::AbstractArray, blockedperm::BlockedPermutation)
  a_perm = _permutedims(a, Tuple(blockedperm))
  return fusedims(a_perm, trivialperm(blockedperm))
end
