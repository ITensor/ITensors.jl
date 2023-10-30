# DiagonalArrays.jl

A Julia `DiagonalArray` type.

````julia
using NDTensors.DiagonalArrays:
  DiagonalArray,
  densearray,
  diagview,
  diaglength,
  getdiagindex,
  setdiagindex!,
  setdiag!,
  diagcopyto!

d = DiagonalArray([1., 2, 3], 3, 4, 5)
@show d[1, 1, 1] == 1
@show d[2, 2, 2] == 2
@show d[1, 2, 1] == 0

d[2, 2, 2] = 22
@show d[2, 2, 2] == 22

@show diaglength(d) == 3
@show densearray(d) == d
@show getdiagindex(d, 2) == d[2, 2, 2]

setdiagindex!(d, 222, 2)
@show d[2, 2, 2] == 222

a = randn(3, 4, 5)
new_diag = randn(3)
setdiag!(a, new_diag)
diagcopyto!(d, a)

@show diagview(a) == new_diag
@show diagview(d) == new_diag
````

You can generate this README with:
```julia
using Literate
Literate.markdown("examples/README.jl", "."; flavor=Literate.CommonMarkFlavor())
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

