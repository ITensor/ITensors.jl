# Similar to `Base.rewrap_unionall` but handles
# more general cases of `TypeVar` parameters.
@generated function to_unionall(type_type::Type)
  type = only(type_type.parameters)
  params = Base.unwrap_unionall(type).parameters
  for i in reverse(eachindex(params))
    param = params[i]
    if param isa TypeVar
      type = UnionAll(param, type)
    end
  end
  return type
end
