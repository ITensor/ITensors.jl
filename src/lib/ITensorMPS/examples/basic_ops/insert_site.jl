using ITensors

function insert(ψ::MPS, b::Integer, ψb::ITensor)
  dataψ′ = insert!(copy(ITensors.data(ψ)), b, ψb)
  ψ′ = MPS(dataψ′)
  lb = linkind(ψ, b - 1)
  l′b = sim(lb)
  δlb = δ(dag(lb), l′b)
  ψ′[b - 1] *= δlb
  ψ′[b] *= dag(δlb)
  return ψ′
end

N = 4
s = siteinds("S=1/2", N; conserve_qns=true)
ψ = randomMPS(s, n -> isodd(n) ? "↑" : "↓")

b = 2
sb = siteind("S=1/2"; conserve_qns=true)
ψb = onehot(sb => 1)
ψ′ = insert(ψ, b, ψb)

@show norm(prod(ψ) * ψb - prod(ψ′))
