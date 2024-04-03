struct TDVPOrder{order,direction} end

TDVPOrder(order::Int, direction::Base.Ordering) = TDVPOrder{order,direction}()

orderings(::TDVPOrder) = error("Not implemented")
sub_time_steps(::TDVPOrder) = error("Not implemented")

function orderings(::TDVPOrder{1,direction}) where {direction}
  return [direction, Base.ReverseOrdering(direction)]
end
sub_time_steps(::TDVPOrder{1}) = [1, 0]

function orderings(::TDVPOrder{2,direction}) where {direction}
  return [direction, Base.ReverseOrdering(direction)]
end
sub_time_steps(::TDVPOrder{2}) = [1 / 2, 1 / 2]

function orderings(::TDVPOrder{4,direction}) where {direction}
  return [direction, Base.ReverseOrdering(direction)]
end
function sub_time_steps(::TDVPOrder{4})
  s = 1 / (2 - 2^(1 / 3))
  return [s / 2, s / 2, (1 - 2 * s) / 2, (1 - 2 * s) / 2, s / 2, s / 2]
end
