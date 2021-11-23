function layered_layout(g)
 xs, ys, _ = solve_positions(Zarate(), g)
 return Point.(zip(xs, ys))
end
