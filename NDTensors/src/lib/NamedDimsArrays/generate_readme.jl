using Literate
using NDTensors.NamedDimsArrays: NamedDimsArrays
Literate.markdown(
  joinpath(
    pkgdir(NamedDimsArrays), "src", "NamedDimsArrays", "examples", "example_readme.jl"
  ),
  joinpath(pkgdir(NamedDimsArrays), "src", "NamedDimsArrays");
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
