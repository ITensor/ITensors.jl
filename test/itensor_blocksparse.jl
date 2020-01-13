using ITensors,
      Test

@testset "BlockSparse ITensor" begin

  i = Index([QN(0)=>1,QN(1)=>2],"i")
  j = Index([QN(0)=>3,QN(-1)=>4,QN(-2)=>5],"j")

  A = ITensor(QN(0),i,j)

  @show A

end

