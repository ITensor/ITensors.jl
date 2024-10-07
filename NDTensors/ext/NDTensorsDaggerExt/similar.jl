using NDTensors: NDTensors
using NDTensors.Expose: Exposed, unexpose
using Dagger: DArray

function NDTensors.similar(E::Exposed{<:DArray})
  A = unexpose(E)
  return Base.similar(A)
end

function similar(E::Exposed{<:DArray}, eltype::Type)
  return Base.similar(unexpose(E), eltype)
end
