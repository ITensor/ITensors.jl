using ITensors
using ITensorUnicodePlots

s = siteinds("S=1/2", 5; conserve_qns=true)
ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=2)
orthogonalize!(ψ, 2)
ψdag = prime(linkinds, dag(ψ))
tn = [ψ..., ψdag...]

edge_labels = (; plevs=true)
@visualize fig tn edge_labels = edge_labels edge_textsize = 20 width = 100

fig
