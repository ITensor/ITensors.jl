# NDTensors.append!
# Used to circumvent issues with some GPU backends like Metal
# not supporting `resize!`.
function append!!(collection, collections...)
  return append!!(unwrap_type(collection), collection, collections...)
end

function append!!(::Type, collection, collections...)
  return append!(collection, collections...)
end
