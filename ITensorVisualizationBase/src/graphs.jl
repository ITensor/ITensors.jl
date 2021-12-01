"""
    Grid

Gride layout.
"""
struct Grid end

(::Grid)(g) = Point.(5 .* (vertices(g) .- 1), 0)

is_self_loop(e::AbstractEdge) = src(e) == dst(e)
any_self_loops(g::AbstractGraph) = any(is_self_loop, edges(g))
