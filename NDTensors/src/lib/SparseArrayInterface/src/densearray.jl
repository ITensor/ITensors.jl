# Generic functionality for converting to a
# dense array, trying to preserve information
# about the array (such as which device it is on).
# TODO: Maybe call `densecopy`?
# TODO: Make sure this actually preserves the device,
# maybe use `NDTensors.Unwrap.unwrap_type`.
function densearray(a::AbstractArray)
  # TODO: `set_ndims(unwrap_type(a), ndims(a))(a)`
  # Maybe define `densetype(a) = set_ndims(unwrap_type(a), ndims(a))`.
  # Or could use `unspecify_parameters(unwrap_type(a))(a)`.
  return Array(a)
end
