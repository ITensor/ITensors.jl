using ITensors: Index, hasinds, permute

function main()
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")

  a = randn(i, j)
  b = randn(j, k)

  @show rand(Int, i, j)
  @show zeros(Float32, i, j)
  @show ones(Float32, i, j)
  @show fill(1.2, i, j)

  a[j => 1, i => 2] = 21
  @show a[2, 1] == 21
  @show a[j => 1, i => 2] == 21

  c = a * b
  @show hasinds(c, (i, k))
  @show permute(a, (j, i))

  # Broken
  a′ = randn(j, i)
  @show a + a′
end

main()
