include("settings.jl")

makedocs(; sitename=sitename, settings...)

deploydocs(;
  repo="github.com/ITensor/ITensors.jl.git",
  devbranch="main",
  push_preview=true,
  deploy_config=Documenter.GitHubActions(),
)
