function nonzero_keys(a::Transpose)
  return (
    CartesianIndex(reverse(Tuple(parent_index))) for
    parent_index in nonzero_keys(parent(a))
  )
end
