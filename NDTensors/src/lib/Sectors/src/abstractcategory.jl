abstract type AbstractCategory end

abstract type AbstractGroup <: AbstractCategory end

label(c::AbstractCategory) = error("method `label` not defined for type $(typeof(c))")

function dimension(c::AbstractCategory)
  error("method `dimension` not defined for category or group $(typeof(c))")
end

function trivial(::Type{T}) where {T<:AbstractCategory}
  error("method `trivial` not defined for category or group $T")
end

function fusion_rule(::Type{C},l1,l2) where {C<:AbstractCategory}
  error("fusion_rule not defined for type $(C)")
end

function fusion_rule(c1::AbstractCategory,c2::AbstractCategory)
  C = typeof(c1)
  return [C(d) for d in fusion_rule(C,label(c1),label(c2))]
end

⊗(c1::AbstractCategory,c2::AbstractCategory) = fusion_rule(c1,c2)

Base.:(*)(s1::AbstractCategory, s2::AbstractCategory) = ⊗(s1, s2)

⊕(a::AbstractCategory, b::AbstractCategory) = [a, b]
⊕(v::Vector{<:AbstractCategory}, b::AbstractCategory) = vcat(v, b)
⊕(a::AbstractCategory, v::Vector{<:AbstractCategory}) = vcat(a, v)

function Base.show(io::IO, q::Vector{<:AbstractCategory})
  isempty(q) && return nothing
  symbol = ""
  for l in q
    print(io, symbol, l)
    symbol = " ⊕ "
  end
end
