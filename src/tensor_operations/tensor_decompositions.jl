"""
Here we create functions which compute the decomposition of a tensor
or a tensor network using higher order tensor decomposition techniques.
I.E the Tucker and CP decompositions

#options
  - threshold : 1e-8; How aggressive the HOSVD scheme is at throwing away small
    valued singular vectors, used to adjust the tucker rank of a mode.
"""

#ITensors.use_debug_checks() = true
## TODO Make it possible to fuse tags together
# before computing the HOSVD

function tucker_HOSVD(T::ITensor; threshold::Float64 = 1e-8)
    number_of_modes = ndims(T) #number of indices in T
    factors = [] # This will have tucker factors stored as itensors
    idx = inds(T)
    # loop through the number of indices in T
    for i = 1:ndims(T)
      # HOSVD first computes the square of the reference
      # This squares the condition number but makes SVD easier
      sq_factor = T * (setprime(T,1, tags=idx[i]))
      D, U = eigen(sq_factor, ishermitian=true,
      lefttags = tags(idx[i]), righttags = addtags(tags(idx[i]), "tucker"), cutoff = threshold)
      T = T * U
      push!(factors, U)
    end
    #push!(factors, T)
    return factors, T
end
