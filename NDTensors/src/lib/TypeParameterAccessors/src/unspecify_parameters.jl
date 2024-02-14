unspecify_parameters(type::DataType) = Base.typename(type).wrapper

function unspecify_parameters(type::UnionAll)
  return unspecify_parameters(Base.unwrap_unionall(type))
end
