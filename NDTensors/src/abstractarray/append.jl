using .TypeParameterAccessors: unwrap_array_type
# NDTensors.append!
# Used to circumvent issues with some GPU backends like Metal
# not supporting `resize!`.
# TODO: Change this over to use `expose`.
function append!!(collection, collections...)
  return append!!(unwrap_array_type(collection), collection, collections...)
end

function append!!(::Type, collection, collections...)
  return append!(collection, collections...)
end
