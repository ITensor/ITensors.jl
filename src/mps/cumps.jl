function cuMPS(psi::MPS)
    phi = copy(psi)
    for site in 1:length(psi)
        phi.A_[site] = cuITensor(psi.A_[site])
    end
    return phi
end
function cuMPS(sites::SiteSet) # random MPS
    N = length(sites)
    MPS(N,fill(cuITensor(),N),0,N+1)
end
function cuMPS(::Type{T}, is::InitState; store_type::DataType=Float64) where {T}
    return cuMPS(MPS(T, is, store_type=store_type))
end
cuMPS(N::Int, d::Int, opcode::String; store_type::DataType=Float64) = cuMPS(InitState(Sites(N,d), opcode), store_type=store_type)
cuMPS(s::SiteSet, opcode::String; store_type::DataType=Float64) = cuMPS(InitState(s, opcode), store_type=store_type)

function randomCuMPS(sites::SiteSet, m::Int=1)
  psi = cuMPS(sites)
  for i=1:length(psi)
    psi[i] = randomCuITensor(sites[i])
    psi[i] /= norm(psi[i])
  end
  if m > 1
    error("randomCuMPS: currently only m==1 supported")
  end
  return psi
end
