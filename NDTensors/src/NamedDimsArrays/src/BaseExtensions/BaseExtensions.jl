module BaseExtensions
replace(collection, replacements::Pair...) = Base.replace(collection, replacements...)
function replace(collection::Tuple, replacements::Pair...)
  if VERSION < v"1.7"
    # TODO: Add to `Compat.jl` or delete when we drop Julia 1.6 support.
    return Tuple(Base.replace(collect(collection), replacements...))
  end
  return Base.replace(collection, replacements...)
end
end
