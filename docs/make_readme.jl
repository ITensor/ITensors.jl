using Literate: Literate
using ITensors: ITensors

Literate.markdown(
  joinpath(pkgdir(ITensors), "examples", "README.jl"),
  joinpath(pkgdir(ITensors));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
