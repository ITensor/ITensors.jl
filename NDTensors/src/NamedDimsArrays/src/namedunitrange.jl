struct NamedUnitRange{T,Value<:AbstractUnitRange{T},Name} <:
       AbstractNamedUnitRange{T,Value,Name}
  value::Value
  name::Name
end

# Interface
unname(i::NamedUnitRange) = i.value
name(i::NamedUnitRange) = i.name

# Constructor
named(i::AbstractUnitRange, name) = NamedUnitRange(i, name)
