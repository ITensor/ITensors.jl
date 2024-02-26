# These definitions help with generic code, where
# we don't know what kind of position will be passed
# but we want to canonicalize to `Position` positions.
position(type::Type, pos::Position) = pos
position(type::Type, pos::Int) = Position(pos)
# Used for named positions.
function position(type::Type, pos)
  return error(
    "Type parameter position not defined for type `$(type)` and position name `$(pos)`."
  )
end
