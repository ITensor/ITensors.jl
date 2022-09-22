"""
Here we create functions which compute the decomposition of a tensor
or a tensor network using higher order tensor decomposition techniques.
I.E the Tucker and CP decompositions
"""

function tucker_HOSVD(T::ITensor; threshold::Float64 = 1e-8)
    number_of_modes = ndims(T) #number of indices in T
    factors = [] # This will have tucker factors stored as itensors
    inds = inds(A)
    # loop through the number of indices in T
    for i = 1:ndims(A)
      # HOSVD first computes the square of the reference
      # This squares the condition number but makes SVD easier
      sq_factor = T * (setprime(T, i, tags=inds[i]))
      SVD = eigen(sq_factor, ishermitian=true, cutoff= 1e-3)
      
    end

end

let
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(10, "k")
  l = Index(8, "l")
  m = Index(18, "m")

  A = randomITensor(i,j,k)
  B = randomITensor(i,k,m,l)
  C = randomITensor(i,m,l,j,k)

  sq_factor = A * (setprime(A, 1, tags=inds(A)[1]))
  @show SVD =
end
