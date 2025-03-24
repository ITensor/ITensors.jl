using Documenter
using ITensors

# Allows using ITensorMPS.jl docstrings in ITensors.jl documentation:
# https://github.com/JuliaDocs/Documenter.jl/issues/1734
DocMeta.setdocmeta!(ITensors, :DocTestSetup, :(using ITensors); recursive=true)

readme_ccq_logo = """<img src="docs/src/assets/CCQ.png" width="20%" alt="Flatiron Center for Computational Quantum Physics logo.">"""
index_ccq_logo = """
  ```@raw html
  <img src="assets/CCQ.png" width="20%" alt="Flatiron Center for Computational Quantum Physics logo.">
  ```
  """

readme_str = read(joinpath(@__DIR__, "..", "README.md"), String)
write(
  joinpath(@__DIR__, "src", "index.md"),
  replace(readme_str, readme_ccq_logo => index_ccq_logo),
)

sitename = "ITensors.jl"

settings = Dict(
  :pages => [
    "Introduction" => "index.md",
    "Getting Started with ITensor" => [
      "Installing Julia and ITensor" => "getting_started/Installing.md",
      "Running ITensor and Julia Codes" => "getting_started/RunningCodes.md",
      "Enabling Debug Checks" => "getting_started/DebugChecks.md",
      "Next Steps" => "getting_started/NextSteps.md",
    ],
    "Code Examples" => ["ITensor Examples" => "examples/ITensor.md"],
    "Documentation" =>
      ["Index" => "IndexType.md", "ITensor" => "ITensorType.md", "QN" => "QN.md"],
    "Frequently Asked Questions" => [
      "Programming Language (Julia, C++, ...) FAQs" => "faq/JuliaAndCpp.md",
      "ITensor Development FAQs" => "faq/Development.md",
      "Julia Package Manager FAQs" => "faq/JuliaPkg.md",
      "High-Performance Computing FAQs" => "faq/HPC.md",
    ],
    "Upgrade guides" => ["Upgrading from 0.1 to 0.2" => "UpgradeGuide_0.1_to_0.2.md"],
    "Advanced Usage Guide" => [
      "Multithreading" => "Multithreading.md",
      "Running on GPUs" => "RunningOnGPUs.md",
      "Contraction sequence optimization" => "ContractionSequenceOptimization.md",
      "HDF5 File Formats" => "HDF5FileFormats.md",
    ],
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none,
)
