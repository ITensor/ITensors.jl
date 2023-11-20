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
