abstract type AbstractCategory end

label(c::AbstractCategory) = error("method `label` not defined for type $(typeof(c))")

function dimension(c::AbstractCategory)
  return error("method `dimension` not defined for type $(typeof(c))")
end

function label_fusion_rule(category_type::Type{<:AbstractCategory}, l1, l2)
  return error("`label_fusion_rule` not defined for type $(category_type).")
end

function fusion_rule(c1::AbstractCategory, c2::AbstractCategory)
  category_type = typeof(c1)
  return [category_type(l) for l in label_fusion_rule(category_type, label(c1), label(c2))]
end

⊗(c1::AbstractCategory, c2::AbstractCategory) = fusion_rule(c1, c2)

⊕(c1::AbstractCategory, c2::AbstractCategory) = [c1, c2]
⊕(cs::Vector{<:AbstractCategory}, c::AbstractCategory) = [cs; c]
⊕(c::AbstractCategory, cs::Vector{<:AbstractCategory}) = [c; cs]

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

function trivial(category_type::Type{<:AbstractCategory})
  return error("`trivial` not defined for type $(category_type).")
end

istrivial(c::AbstractCategory) = (c == trivial(typeof(c)))

function dual(category_type::Type{<:AbstractCategory})
  return error("`dual` not defined for type $(category_type).")
end

Base.isless(c1::AbstractCategory, c2::AbstractCategory) = isless(label(c1), label(c2))
