abstract type AbstractCategory end

label(c::AbstractCategory) = error("method `label` not defined for type $(typeof(c))")

function dimension(c::AbstractCategory)
  return error("method `dimension` not defined for type $(typeof(c))")
end

function label_fusion_rule(::Type{C}, l1, l2) where {C<:AbstractCategory}
  return error("label_fusion_rule not defined for type $(C)")
end

function fusion_rule(c1::AbstractCategory, c2::AbstractCategory)
  C = typeof(c1)
  return [C(d) for d in label_fusion_rule(C, label(c1), label(c2))]
end

⊗(c1::AbstractCategory, c2::AbstractCategory) = fusion_rule(c1, c2)

Base.:(*)(s1::AbstractCategory, s2::AbstractCategory) = ⊗(s1, s2)

⊕(a::AbstractCategory, b::AbstractCategory) = [a, b]
⊕(v::Vector{<:AbstractCategory}, b::AbstractCategory) = vcat(v, b)
⊕(a::AbstractCategory, v::Vector{<:AbstractCategory}) = vcat(a, v)

function Base.show(io::IO, q::Vector{<:AbstractCategory})
  (length(q) <= 1) && print(io, "[")
  symbol = ""
  for l in q
    print(io, symbol, l)
    symbol = " ⊕ "
  end
  return (length(q) <= 1) && print(io, "]")
end

function trivial(::Type{C}) where {C<:AbstractCategory}
  return error("method `trivial` not defined for type $C")
end

istrivial(C::AbstractCategory) = (C == trivial(typeof(C)))
