struct Position{Pos} end
Position(pos) = Position{pos}()
Base.Int(pos::Position) = Int(parameter(typeof(pos)))
