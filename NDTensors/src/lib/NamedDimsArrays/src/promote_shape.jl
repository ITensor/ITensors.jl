function Base.promote_shape(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return promote_shape(namedaxes(na1), namedaxes(na2))
end

function Base.promote_shape(
  na1::Tuple{Vararg{AbstractNamedUnitRange}}, na2::Tuple{Vararg{AbstractNamedUnitRange}}
)
  a1 = unname(na1)
  a2 = unname(na2, dimnames(na1))
  a_promoted = promote_shape(a1, a2)
  return named(a_promoted, dimnames(na1))
end
