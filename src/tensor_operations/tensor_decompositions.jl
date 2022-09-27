"""
Here we create functions which compute the decomposition of a tensor
or a tensor network using higher order tensor decomposition techniques.
I.E the Tucker and CP decompositions

#options
  - threshold : 1e-8; How aggressive the HOSVD scheme is at throwing away small
    valued singular vectors, used to adjust the tucker rank of a mode.

return list of factor matrices and the new core tensor as a tuple
"""

#ITensors.use_debug_checks() = true
## TODO Make it possible to fuse tags together
# before computing the HOSVD

function tucker_hosvd(T::ITensor; threshold::Float64 = 1e-8)
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

function tucker_hosvd!(T::ITensor; threshold::Float64 = 1e-8)
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
    @show inds(T)
    return factors
end

function check_norm(norm_ref::Float64, mttkrp::ITensor,
  factors::Array{ITensor}; verbose::Bool = false)
  # Matrized tensor times khatri rao product
  number_factors = size(factors)
  mttkrp .* factors[number_factors]
  inner_prod = norm(mttkrp)

  #TODO use an iterator

  gram = factor[1] * setprime(factors[1], 1, inds(factors[1])[1])
  for i = 2 : number_factors
    gram .*= factor[i] * setprime(factors[i], 1, inds(factors[i])[1])
  end
  norm_cp = norm(gram)
  norm_residual = sqrt(abs(norm_ref * norm_ref + norm_cp * norm_cp - 2.0 * inner_prod))

  curr_fit = 1.0 - (norm_residual / norm_ref)
  fit_change = abs(curr_fit - prev_fit)

  if verbose
    println(curr_fit, "\t", fit_change);
  fit_change
end

function cp_als(ref::ITensor, rank::Int64; kwargs...)
  als_thresh::FLoat64 = get(kwargs, :als_thresh, 1e-3)
  max_iters::Int64 = get(kwargs, :max_iters, 1000)
  checker::Function = get(kwargs, :checker, check_norm)

  norm_ref = norm(ref)
  factors=
end

let
  i = Index(20, "i")
  j = Index(3, "j")
  k = Index(10, "k")
  l = Index(8, "l")
  m = Index(18, "m")

  A = randomITensor(i,j,k)

  r = Index(30, "rank")
  r2 = Index(40, "test")
  cp_facs = []
  for i in inds(A)
    push!(cp_facs, randomITensor(r,i, r2))
  end

  normA = norm(A)

  # Working on making the khatri_rao_product (general version)
  #Find matching inds
  a = cp_facs[1]
  b = cp_facs[2]
  comm = commoninds(a,b)
  uniq_a = uniqueinds(a,comm)
  uniq_b = uniqueinds(b, comm)

  ninds_a = vcat(comm, uniq_a)
  ninds_b = vcat(comm, uniq_b)
  final_inds = vcat(comm, uniq_a, uniq_b)

  a = permute(a, ninds_a)
  b = permute(b, ninds_b)

  size_of_common = 1
  for i = comm[:]
    size_of_common *= dim(i)
  end
  size_outer_a = 1
  size_outer_b = 1
  for i = uniq_a[:]
    size_outer_a *= dim(i)
  end

  for i = uniq_b[:]
    size_outer_b *= dim(i)
  end
  final_size = size_outer_a * size_outer_b

  result = ITensor(0, final_inds)
  storage_a = storage(a)
  storage_b = storage(b)
  for i = range(1,size_of_common-1, step = 1)
    start = i * size_outer_a
    e = (i + 1) * size_outer_a
    viewA = view(storage_a, [start:e])
    #@show size(viewA)
  end
end
