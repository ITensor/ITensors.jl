abstract type AbstractPosition end

struct Position{Pos} <: AbstractPosition end
Position(pos) = Position{pos}()
Base.Int(pos::Position) = Int(parameter(typeof(pos)))

struct UndefinedPosition <: AbstractPosition end
