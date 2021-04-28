using Documenter, ITensors

DocMeta.setdocmeta!(ITensors, :DocTestSetup, :(using ITensors); recursive=true)

sitename = "ITensors.jl"

settings = Dict(
  :modules => [ITensors],
  :pages => [
    "Introduction" => "index.md",
    "Code Examples" => [
      "ITensor Examples" => "examples/ITensor.md",
      "MPS and MPO Examples" => "examples/MPSandMPO.md",
      "DMRG Examples" => "examples/DMRG.md",
      "Physics System Examples" => "examples/Physics.md",
    ],
    "Documentation" => [
      "Index" => "IndexType.md",
      "IndexVal" => "IndexValType.md",
      "IndexSet" => "IndexSetType.md",
      "ITensor" => "ITensorType.md",
      "MPS and MPO" => "MPSandMPO.md",
      "QN" => "QN.md",
      "SiteType and op" => "SiteType.md",
      "DMRG" => [
        "DMRG.md",
        "Sweeps.md",
        "ProjMPO.md",
        "ProjMPOSum.md",
        "Observer.md",
        "DMRGObserver.md",
      ],
      "AutoMPO" => "AutoMPO.md",
    ],
    "Advanced usage guide" => [
      "Advanced usage guide" => "AdvancedUsageGuide.md",
      "Multithreading" => "Multithreading.md",
      "Symmetric tensor background and usage" => "QNTricks.md",
      "Timing and profiling" => "CodeTiming.md",
      "Contraction sequence optimization" => "ContractionSequenceOptimization.md",
    ],
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none,
)
