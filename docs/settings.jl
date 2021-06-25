using Documenter, ITensors

DocMeta.setdocmeta!(ITensors, :DocTestSetup, :(using ITensors); recursive=true)

sitename = "ITensors.jl"

settings = Dict(
  :modules => [ITensors],
  :pages => [
    "Introduction" => "index.md",
    "Getting Started with ITensor" => [
      "Installing Julia and ITensor" => "getting_started/Installing.md",
      "Running ITensor and Julia Codes" => "getting_started/RunningCodes.md",
      "Tutorials" => "getting_started/Tutorials.md",
      "Next Steps" => "getting_started/NextSteps.md",
    ],
    "Code Examples" => [
      "ITensor Examples" => "examples/ITensor.md",
      "MPS and MPO Examples" => "examples/MPSandMPO.md",
      "DMRG Examples" => "examples/DMRG.md",
      "Physics (SiteType) System Examples" => "examples/Physics.md",
    ],
    "Documentation" => [
      "Index" => "IndexType.md",
      "Index collections" => "IndexSetType.md",
      "ITensor" => "ITensorType.md",
      "MPS and MPO" => "MPSandMPO.md",
      "QN" => "QN.md",
      "SiteType and op, state, val functions" => "SiteType.md",
      "DMRG" => [
        "DMRG.md",
        "Sweeps.md",
        "ProjMPO.md",
        "ProjMPOSum.md",
        "Observer.md",
        "DMRGObserver.md",
      ],
      "OpSum (AutoMPO)" => "OpSum.md",
    ],
    "Upgrade guides" => ["Upgrading from 0.1 to 0.2" => "UpgradeGuide_0.1_to_0.2.md"],
    "ITensor indices and Einstein notation" => "Einsum.md",
    "Advanced usage guide" => [
      "Advanced usage guide" => "AdvancedUsageGuide.md",
      "Multithreading" => "Multithreading.md",
      "Symmetric (QN conserving) tensors: background and usage" => "QNTricks.md",
      "Timing and profiling" => "CodeTiming.md",
      "Contraction sequence optimization" => "ContractionSequenceOptimization.md",
    ],
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none,
)
