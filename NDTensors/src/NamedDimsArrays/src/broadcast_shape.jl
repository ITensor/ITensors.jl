using Base.Broadcast: Broadcast, broadcast_shape, combine_axes

# TODO: Have `axes` output named axes so Base functions "just work".
function Broadcast.combine_axes(na1::AbstractNamedDimsArray, nas::AbstractNamedDimsArray...)
  return broadcast_shape(namedaxes(na1), combine_axes(nas...))
end
function Broadcast.combine_axes(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return broadcast_shape(namedaxes(na1), namedaxes(na2))
end
Broadcast.combine_axes(na::AbstractNamedDimsArray) = namedaxes(na)

function Broadcast.broadcast_shape(
  na1::Tuple{Vararg{AbstractNamedUnitRange}},
  na2::Tuple{Vararg{AbstractNamedUnitRange}},
  nas::Tuple{Vararg{AbstractNamedUnitRange}}...,
)
  return broadcast_shape(broadcast_shape(shape, shape1), shapes...)
end

function Broadcast.broadcast_shape(
  na1::Tuple{Vararg{AbstractNamedUnitRange}}, na2::Tuple{Vararg{AbstractNamedUnitRange}}
)
  return promote_shape(na1, na2)
end
