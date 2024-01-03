@eval module $(gensym())
for filename in ["abstractsparsearray", "array", "diagonalarray"]
  include("test_$filename.jl")
end
end
