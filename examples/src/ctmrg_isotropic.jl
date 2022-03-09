using ITensors

function ctmrg(T::ITensor, Cₗᵤ::ITensor, Aₗ::ITensor; χmax::Int, cutoff=0.0, nsteps::Int)
  sₕ = commonind(T, Aₗ)
  sᵥ = uniqueind(T, Aₗ, Aₗ'; plev=0)
  lᵥ = commonind(Cₗᵤ, Aₗ)
  lₕ = uniqueind(Cₗᵤ, Aₗ)
  Aᵤ = replaceinds(Aₗ, lᵥ => lₕ, lᵥ' => lₕ', sₕ => sᵥ)
  Cₗᵤ = dense(Cₗᵤ)
  for i in 1:nsteps
    ## Get the grown corner transfer matrix (CTM)
    Cₗᵤ⁽¹⁾ = Aₗ * Cₗᵤ * Aᵤ * T

    ## Diagonalize the grown CTM
    # TODO: replace with
    # eigen(Cₗᵤ⁽¹⁾, "horiz" => "vert"; tags = "horiz" => "vert", kwargs...)
    Cₗᵤ, Uᵥ = eigen(
      Cₗᵤ⁽¹⁾,
      (lₕ', sₕ'),
      (lᵥ', sᵥ');
      ishermitian=true,
      cutoff,
      maxdim=χmax,
      lefttags=tags(lₕ),
      righttags=tags(lᵥ),
    )
    Cₗᵤ = dense(Cₗᵤ)
    lᵥ = commonind(Cₗᵤ, Uᵥ)
    lₕ = uniqueind(Cₗᵤ, Uᵥ)

    # The renormalized CTM is the diagonal matrix of eigenvalues
    # Normalize the CTM
    Cₗ = Cₗᵤ * prime(dag(Cₗᵤ), lₕ)
    normC = (Cₗ * dag(Cₗ))[]^(1 / 4)
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
