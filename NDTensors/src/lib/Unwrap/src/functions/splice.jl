function Base.splice!(e::Exposed, indices, replacement...)
  a = unexpose(e)
  res = splice!(a, indices, replacement...)
  return res
end
