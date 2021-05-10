
#
# blas_get_num_threads()
# Get the number of BLAS threads
# This can be replaced by BLAS.get_num_threads() in Julia v1.6
#

function guess_vendor()
  # like determine_vendor, but guesses blas in some cases
  # where determine_vendor returns :unknown
  ret = BLAS.vendor()
  if Sys.isapple() && (ret == :unknown)
    ret = :osxblas
  end
  return ret
end

_tryparse_env_int(key) = tryparse(Int, get(ENV, key, ""))

blas_get_num_threads()::Union{Int,Nothing} = _get_num_threads()

function _get_num_threads(; _blas=guess_vendor())::Union{Int,Nothing}
  if _blas === :openblas || _blas === :openblas64
    return Int(ccall((BLAS.@blasfunc(openblas_get_num_threads), BLAS.libblas), Cint, ()))
  elseif _blas === :mkl
    return Int(ccall((:mkl_get_max_threads, BLAS.libblas), Cint, ()))
  elseif _blas === :osxblas
    key = "VECLIB_MAXIMUM_THREADS"
    nt = _tryparse_env_int(key)
    if nt === nothing
      @warn "Failed to read environment variable $key" maxlog = 1
    else
      return nt
    end
  else
    @assert _blas === :unknown
  end
  @warn "Could not get number of BLAS threads. Returning `nothing` instead." maxlog = 1
  return nothing
end
