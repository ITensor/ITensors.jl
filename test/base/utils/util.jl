using ITensors
using Random

# Based on https://discourse.julialang.org/t/lapackexception-1-while-svd-but-not-svdvals/23787
function make_illconditioned_matrix(T=5000)
  t = 0:(T - 1)
  f = LinRange(0, 0.5 - 1 / length(t) / 2, length(t) ÷ 2)
  y = sin.(t)
  function check_freq(f)
    zerofreq = findfirst(iszero, f)
    zerofreq !== nothing &&
      zerofreq != 1 &&
      throw(ArgumentError("If zero frequency is included it must be the first frequency"))
    return zerofreq
  end
  function get_fourier_regressor(t, f)
    zerofreq = check_freq(f)
    N = length(t)
    Nf = length(f)
    Nreg = zerofreq === nothing ? 2Nf : 2Nf - 1
    N >= Nreg || throw(ArgumentError("Too many frequency components $Nreg > $N"))
    A = zeros(N, Nreg)
    sinoffset = Nf
    for fn in 1:Nf
      if fn == zerofreq
        sinoffset = Nf - 1
      end
      for n in 1:N
        phi = 2π * f[fn] * t[n]
        A[n, fn] = cos(phi)
        if fn != zerofreq
          A[n, fn + sinoffset] = -sin(phi)
        end
      end
    end
    return A, zerofreq
  end
  A, z = get_fourier_regressor(t, f)
  return [A y]
end
