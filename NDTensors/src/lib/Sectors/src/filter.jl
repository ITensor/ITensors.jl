# https://discourse.julialang.org/t/compile-time-type-filtering-from-a-tuple-is-it-possible/101090/2

# Length zero
filtered(::Type{C}, ::Tuple{}) where {C} = ()

# Length one
filtered(::Type{C}, cats::T) where {C,T<:Tuple{C}} = cats
filtered(::Type{C}, cats::T) where {C,T<:Tuple{Any}} = ()

# Length two or more
function filtered(::Type{C}, cats::T) where {C,T<:Tuple{Any,Any,Vararg{Any}}}
  return (filtered(C, (first(cats),))..., filtered(C, Base.tail(cats))...)
end

sieve(c::@NamedTuple) = (cats=categories(c)filtered(B, cats), filtered(Consonant, cats))

function f(
  cats1::NT, cats2::NT
) where {NT<:NamedTuple{<:Any,<:Tuple{AbstractCategory,Vararg{AbstractCategory}}}}
  return println(NT)
end

function gg(
  cats1::NT, cats2::NT
) where {NT<:NamedTuple{<:Any,<:Tuple{AbstractCategory,Vararg{AbstractCategory}}}}
  k = first(keys(cats1))
  return cats1[k]
end

function hh(
  cats1::NT, cats2::NT
) where {Names,NT<:NamedTuple{Names,<:Tuple{AbstractCategory,Vararg{AbstractCategory}}}}
  println(typeof(cats1), " ", Names)
  return fusion_rule(cats1[(Names[1],)], cats2[(Names[1],)]),
  (cats1[(Names[2:end],)], cats2[(Names[2:end],)])
end
