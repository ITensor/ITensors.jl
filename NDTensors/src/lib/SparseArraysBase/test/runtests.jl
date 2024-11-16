@eval module $(gensym())
for filename in ["sparsearraydok", "abstractsparsearray", "array", "diagonalarray"]
  include("test_$filename.jl")
end
end
