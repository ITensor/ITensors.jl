using LinearAlgebra: LinearAlgebra, svd
using ...NDTensors.RankFactorization: Spectrum, truncate!!
function LinearAlgebra.svd(
  na::AbstractNamedDimsArray;
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  alg=nothing,
  min_blockdim=nothing,
)
  # TODO: Handle array wrappers around
  # `AbstractNamedDimsArray` more elegantly.
  USV = svd(unname(na))
  u, s, v = USV.U, USV.S, USV.Vt

  # Sort by largest to smallest eigenvalues
  # TODO: Replace `cpu` with `Expose` dispatch.
  p = sortperm(s; rev=true, by=abs)
  u = u[:, p]
  s = s[p]
  v = v[p, :]

  s² = s .^ 2
  length_s = length(s)
  truncerr = zero(Float64) # Make more generic
  if any(!isnothing, (maxdim, cutoff))
    s², truncerr, _ = truncate!!(
      s²; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    length_s = length(s²)
    # TODO: Avoid this if they are already the
    # correct size.
    u = u[:, 1:length_s]
    s = s[1:length_s]
    v = v[1:length_s, :]
  end
  spec = Spectrum(s², truncerr)

  # TODO: Handle array wrappers more generally.
  names_a = dimnames(na)
  # TODO: Make this more generic, handle `dag`, etc.
  l = randname(names_a[1]) # IndexID(rand(UInt64), "", 0)
  r = randname(names_a[2]) # IndexID(rand(UInt64), "", 0)
  names_u = (names_a[1], l)
  nu = named(u, names_u)
  names_s = (l, r)
  ns = named(Diagonal(s), names_s)
  names_v = (r, names_a[2])
  nv = named(v, names_v)
  return nu, ns, nv, spec
end
