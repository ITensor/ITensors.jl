# Custom version of `sort` (`SmallVectors.sort`) that directly uses an `order::Ordering`.
function sort(v, order::Base.Sort.Ordering; alg::Base.Sort.Algorithm=Base.Sort.defalg(v))
  mv = thaw(v)
  SmallVectors.sort!(mv, order; alg)
  return freeze(mv)
end

# Custom version of `sort!` (`SmallVectors.sort!`) that directly uses an `order::Ordering`.
function sort!(
  v::AbstractVector{T},
  order::Base.Sort.Ordering;
  alg::Base.Sort.Algorithm=Base.Sort.defalg(v),
  scratch::Union{Vector{T},Nothing}=nothing,
) where {T}
  if VERSION < v"1.9"
    Base.sort!(v, alg, order)
  else
    Base.Sort._sort!(
      v, Base.Sort.maybe_apply_initial_optimizations(alg), order, (; scratch)
    )
  end
  return v
end
