parameter(type::Type, pos) = parameter(type, position(type, pos))
parameter(type::Type, pos::Position) = parameters(type)[Int(pos)]
parameter(type::Type) = only(parameters(type))
