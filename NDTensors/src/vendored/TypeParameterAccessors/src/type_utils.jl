"replace `Symbol`s with `QuoteNode`s to avoid expression interpolation"
wrap_symbol_quotenode(param) = param isa Symbol ? QuoteNode(param) : param

"Construct the expression for qualifying a type with given parameters"
function construct_type_expr(type, parameters)
  basetype = unspecify_type_parameters(type)
  type_expr = Expr(:curly, basetype, wrap_symbol_quotenode.(parameters)...)
  for parameter in reverse(parameters)
    if parameter isa TypeVar
      type_expr = Expr(:call, :UnionAll, parameter, type_expr)
    end
  end
  return type_expr
end
