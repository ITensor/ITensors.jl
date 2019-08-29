
function isdisjoint(s1,s2)
  for i1 ∈ length(s1)
    for i2 ∈ length(s2)
      s1[i1] == s2[i2] && return false
    end
  end
  return true
end

