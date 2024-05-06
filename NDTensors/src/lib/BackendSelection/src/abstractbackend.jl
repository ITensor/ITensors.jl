abstract type AbstractBackend end

backend_string(::AbstractBackend) = error("Not implemented")
parameters(::AbstractBackend) = error("Not implemented")
