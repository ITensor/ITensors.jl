function specify_parameter(type::Type, pos, param)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, pos, param)
end

for f in [:set_parameter, :specify_parameter]
  fs = Symbol(f, :s)
  _fs = Symbol(:_, f, :s)
  @eval begin
    function $_fs(type::Type, positions::Tuple{Vararg{Int}}, params::Tuple)
      @assert length(positions) == length(params)
      for i in 1:length(positions)
        type = $f(type, positions[i], params[i])
      end
      return type
    end
    @generated function $fs(
      type_type::Type,
      positions_type::Tuple{Vararg{Position}},
      params_type::Tuple{Vararg{TypeParameter}},
    )
      type = parameter(type_type)
      positions = parameter.(parameters(positions_type))
      params = parameter.(parameters(params_type))
      return $_fs(type, positions, params)
    end
    function $fs(type::Type, positions::Tuple, params::Tuple)
      return $fs(type, position.(type, positions), TypeParameter.(params))
    end
    function $fs(type::Type, params::Tuple)
      return $fs(type, eachposition(type), params)
    end
  end
end
