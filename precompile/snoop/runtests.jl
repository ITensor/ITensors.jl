
# Many test files are leading
# to precompile command that
# cause errors, so they
# are commented out

for filename in (
  "tagset.jl",
  "smallstring.jl",
  "index.jl",
  "indexset.jl",
  "not.jl",
  #"itensor_dense.jl",
  #"itensor_diag.jl",
  #"contract.jl",
  #"combiner.jl",
  #"trg.jl",
  #"ctmrg.jl",
  #"iterativesolvers.jl",
  #"dmrg.jl",
  #"tag_types.jl",
  #"phys_site_types.jl",
  #"decomp.jl",
  #"lattices.jl",
  #"mps.jl",
  #"mpo.jl",
  #"autompo.jl",
  #"svd.jl",
  #"qn.jl",
  #"qnindex.jl",
  #"itensor_blocksparse.jl",
  #"itensor_diagblocksparse.jl",
  #"readwrite.jl",
  #"readme.jl",
  #"examples.jl",
)
  include("../../test/$filename")
end
