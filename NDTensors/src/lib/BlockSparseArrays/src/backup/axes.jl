# TODO: Delete
## function NDTensors.outer(s1::Base.OneTo, s2::Base.OneTo)
##   return Base.OneTo(length(s1) * length(s2))
## end

function blockmerge(s::Base.OneTo, grouped_perm::Vector{Vector{Int}})
  @assert grouped_perm == [[1]]
  return s
end

blockmergesortperm(s::Base.OneTo) = [[1]]

function sub_axis(a::BlockedUnitRange, blocks)
  return blockedrange([length(a[b]) for b in blocks])
end

function sub_axes(axes_src::Tuple, axes_parent::Tuple)
  return sub_axis.(axes_src, axes_parent)
end
