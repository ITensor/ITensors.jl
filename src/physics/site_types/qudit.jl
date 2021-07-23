
ITensors.space(::SiteType"Qudit"; dim=2) = dim

function ITensors.state(::StateName{N}, ::SiteType"Qudit", s::Index) where {N}
  n = parse(Int, String(N))
  st = zeros(dim(s))
  st[n + 1] = 1.0
  return st
end
