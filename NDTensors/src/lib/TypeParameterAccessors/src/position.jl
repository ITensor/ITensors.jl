## Position
struct Position{Pos} end
Position(pos) = Position{pos}()
Base.Int(pos::Position) = parameter(pos)

## position
# These definitions help with generic code, where
# we don't know what kind of position will be passed
# but we want to canonicalize to `Position` positions.
position(type::Type, pos::Int) = Position(pos)
position(type::Type, pos::Position) = pos

## eachposition
eachposition(type::Type) = ntuple(Position, Val(nparameters(type)))

position_name(type::Type) = Base.Fix1(position_name, type)
position_names(type::Type) = map(position_name(type), eachposition(type))
