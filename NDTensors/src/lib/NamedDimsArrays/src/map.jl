# TODO: Handle maybe-mutation.
# TODO: Handle permutations more efficiently by fusing with `f`.
function Base.map!(f, na_dest::AbstractNamedDimsArray, nas::AbstractNamedDimsArray...)
  a_dest = unname(na_dest)
  as = map(na -> unname(na, dimnames(na_dest)), nas)
  map!(f, a_dest, as...)
  return na_dest
end

function Base.map(f, nas::AbstractNamedDimsArray...)
  na_dest = similar(first(nas))
  map!(f, na_dest, nas...)
  return na_dest
end
