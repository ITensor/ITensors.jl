using ITensors

function ctmrg(T::ITensor,
               Cₗᵤ::ITensor,
               Aₗ::ITensor;
               χmax::Int, nsteps::Int)
  sₕ = commonind(T, Aₗ)
  sᵥ = firstind(noncommoninds(T, Aₗ, Aₗ'); plev = 0)
  lᵥ = commonind(Cₗᵤ, Aₗ)
  # TODO: use noncommonind when fixed
  lₕ = noncommoninds(Cₗᵤ, Aₗ)[1]
  Aᵤ = replaceinds(Aₗ, lᵥ => lₕ, lᵥ' => lₕ', sₕ => sᵥ)
  for i in 1:nsteps
    ## Get the grown corner transfer matrix (CTM)
    Cₗᵤ⁽¹⁾ = Aₗ * Cₗᵤ * Aᵤ * T

    ## Diagonalize the grown CTM
    Cₗᵤ, Uᵥ = eigen(Cₗᵤ⁽¹⁾, (lₕ', sₕ'), (lᵥ', sᵥ');
                    ishermitian = true,
                    maxdim = χmax,
                    lefttags = tags(lₕ),
                    righttags = tags(lᵥ))

    lᵥ = commonind(Cₗᵤ, Uᵥ)
    # TODO: use noncommonind when fixed
    lₕ = noncommoninds(Cₗᵤ, Uᵥ)[1]

    # The renormalized CTM is the diagonal matrix of eigenvalues
    # Normalize the CTM
    Cₗ = Cₗᵤ * prime(dag(Cₗᵤ), lₕ)
    normC = (Cₗ * dag(Cₗ))[] ^ (1 / 4)
    Cₗᵤ = Cₗᵤ / normC

    # Calculate the renormalized half row transfer matrix (HRTM)
    Uᵥ = noprime(Uᵥ)
    Aₗ = Aₗ * Uᵥ * T * dag(Uᵥ')
    Aₗ = replaceinds(Aₗ, sₕ' => sₕ)

    # Normalize the HRTM
    ACₗ = Aₗ * Cₗᵤ * prime(dag(Cₗᵤ))
    normA = √((ACₗ * dag(ACₗ))[])
    Aₗ = Aₗ / normA
    Aᵤ = replaceinds(Aₗ, lᵥ => lₕ, lᵥ' => lₕ', sₕ => sᵥ)
  end
  return Cₗᵤ, Aₗ
end

