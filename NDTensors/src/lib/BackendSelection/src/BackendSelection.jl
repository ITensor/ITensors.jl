module BackendSelection
include("abstractbackend.jl")
include("backend_types.jl")

# TODO: This is defined for backwards compatibility,
# delete this alias once downstream packages change over
# to using `BackendSelection`.
const AlgorithmSelection = BackendSelection
end
