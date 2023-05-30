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
      "Enabling Debug Checks" => "getting_started/DebugChecks.md",
      "Next Steps" => "getting_started/NextSteps.md",
    ],
    "Tutorials" => [
      "DMRG" => "tutorials/DMRG.md",
      "Quantum Number Conserving DMRG" => "tutorials/QN_DMRG.md",
      "MPS Time Evolution" => "tutorials/MPSTimeEvolution.md",
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
      "SiteTypes Included with ITensor" => "IncludedSiteTypes.md",
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
    "Frequently Asked Questions" => [
      "Programming Language (Julia, C++, ...) FAQs" => "faq/JuliaAndCpp.md",
      "DMRG FAQs" => "faq/DMRG.md",
      "Quantum Number (QN) FAQs" => "faq/QN.md",
      "ITensor Development FAQs" => "faq/Development.md",
      "Relationship of ITensor to other tensor libraries FAQs" => "faq/RelationshipToOtherLibraries.md",
      "Julia Package Manager FAQs" => "faq/JuliaPkg.md",
    ],
    "Upgrade guides" => ["Upgrading from 0.1 to 0.2" => "UpgradeGuide_0.1_to_0.2.md"],
    "ITensor indices and Einstein notation" => "Einsum.md",
    "Advanced Usage Guide" => [
      "Advanced Usage Guide" => "AdvancedUsageGuide.md",
      "Multithreading" => "Multithreading.md",
      "Symmetric (QN conserving) tensors: background and usage" => "QNTricks.md",
      "Timing and profiling" => "CodeTiming.md",
      "Contraction sequence optimization" => "ContractionSequenceOptimization.md",
      "HDF5 File Formats" => "HDF5FileFormats.md",
    ],
    "Developer Guide" => "DeveloperGuide.md",
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none,
)
