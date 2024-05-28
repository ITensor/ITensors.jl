default_maxdim() = typemax(Int)
default_mindim() = 1
default_cutoff(type::Type{<:Number}) = eps(real(type))
default_noise() = false
