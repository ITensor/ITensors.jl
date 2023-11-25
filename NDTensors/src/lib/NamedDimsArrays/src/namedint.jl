struct NamedInt{Value,Name} <: AbstractNamedInt{Value,Name}
  value::Value
  name::Name
end

## Needs a `default_name(nametype::Type)` function.
## NamedInt{Value,Name}(i::Integer) where {Value,Name} = NamedInt{Value,Name}(i, default_name(Name))

# Interface
unname(i::NamedInt) = i.value
name(i::NamedInt) = i.name

# Convenient constructor
named(i::Integer, name) = NamedInt(i, name)

# TODO: Use `isnamed` trait?
dimnames(a::Tuple{Vararg{AbstractNamedInt}}) = name.(a)

function get_name_perm(a::Tuple{Vararg{AbstractNamedInt}}, names::Tuple)
  return getperm(dimnames(a), names)
end

# Permute into a certain order.
# align(a, (:j, :k, :i))
function align(a::Tuple{Vararg{AbstractNamedInt}}, names)
  perm = get_name_perm(a, names)
  return map(j -> a[j], perm)
end
