using Literate: Literate
using ITensors: ITensors

Literate.markdown(
  joinpath(pkgdir(ITensors), "examples", "README.jl"),
  joinpath(pkgdir(ITensors), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
