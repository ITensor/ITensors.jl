# Use like this:
#
# using LayeredLayouts
# layout = layered_layouts(solve_positions(Zarate(), g))
#
function layered_layout(pos)
  xs, ys, _ = pos
  return Point.(zip(xs, ys))
end
