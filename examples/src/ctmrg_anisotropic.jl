using ITensors

function site_inds(ny, nx, d=1)
  sh = Matrix{Index}(undef, ny, nx)
  sv = Matrix{Index}(undef, ny, nx)
  for iy in 1:ny, ix in 1:nx
    sh[iy, ix] = Index(d, "site,horiz,x=$ix,y=$iy")
    sv[iy, ix] = Index(d, "site,vert,x=$ix,y=$iy")
  end
  return sh, sv
end

function link_inds(ny, nx, d=1)
  ll = Matrix{Index}(undef, ny, nx)
  lr = Matrix{Index}(undef, ny, nx)
  lu = Matrix{Index}(undef, ny, nx)
  ld = Matrix{Index}(undef, ny, nx)
  for iy in 1:ny, ix in 1:nx
    ll[iy, ix] = Index(d, "link,left,x=$ix,y=$iy")
    lr[iy, ix] = Index(d, "link,right,x=$ix,y=$iy")
    lu[iy, ix] = Index(d, "link,up,x=$ix,y=$iy")
    ld[iy, ix] = Index(d, "link,down,x=$ix,y=$iy")
  end
  return ll, lr, lu, ld
end

per(n, N) = mod(n - 1, N) + 1

function ctmrg_environment((sh, sv))
  ny, nx = size(sh)
  Clu = Matrix{ITensor}(undef, ny, nx)
  Cru = Matrix{ITensor}(undef, ny, nx)
  Cld = Matrix{ITensor}(undef, ny, nx)
  Crd = Matrix{ITensor}(undef, ny, nx)

  Al = Matrix{ITensor}(undef, ny, nx)
  Ar = Matrix{ITensor}(undef, ny, nx)
  Au = Matrix{ITensor}(undef, ny, nx)
  Ad = Matrix{ITensor}(undef, ny, nx)

  ll, lr, lu, ld = link_inds(ny, nx)
  for iy in 1:ny, ix in 1:nx
    Clu[iy, ix] = randomITensor(ll[iy, ix], lu[iy, ix])
    Cru[iy, ix] = randomITensor(lr[iy, ix], lu[iy, ix])
    Cld[iy, ix] = randomITensor(ll[iy, ix], ld[iy, ix])
    Crd[iy, ix] = randomITensor(lr[iy, ix], ld[iy, ix])
    iyp, ixp = per(iy + 1, ny), per(ix + 1, nx)
    Al[iy, ix] = randomITensor(sh[iy, ix], ll[iy, ix], ll[iyp, ix])
    Ar[iy, ix] = randomITensor(sh[iy, ix], lr[iy, ix], lr[iyp, ix])
    Au[iy, ix] = randomITensor(sv[iy, ix], lu[iy, ix], lu[iy, ixp])
    Ad[iy, ix] = randomITensor(sv[iy, ix], ld[iy, ix], ld[iy, ixp])
  end
  normalize!((Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))
  return (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad)
end

function calc_κ(iy, ix, T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir="left")
  ny, nx = size(T)
  iyp, ixp = per(iy + 1, ny), per(ix + 1, nx)
  normC = scalar(Clu[iy, ix] * Cld[iy, ix] * Cru[iy, ix] * Crd[iy, ix])
  normAlr = scalar(
    Clu[iy, ix] * Cru[iy, ix] * Al[iy, ix] * Ar[iy, ix] * Cld[iyp, ix] * Crd[iyp, ix]
  )
  normAud = scalar(
    Clu[iy, ix] * Cld[iy, ix] * Au[iy, ix] * Ad[iy, ix] * Cru[iy, ixp] * Crd[iy, ixp]
  )
  normT = scalar(
    Clu[iy, ix] *
    Al[iy, ix] *
    Cld[iyp, ix] *
    Au[iy, ix] *
    T[iy, ix] *
    Ad[iyp, ix] *
    Cru[iy, ixp] *
    Ar[iy, ixp] *
    Crd[iyp, ixp],
  )
  return normT, normAlr, normAud, normC
end

function calc_κ(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir="left")
  ny, nx = size(T)
  κ = Matrix{Float64}(undef, ny, nx)
  for iy in 1:ny, ix in 1:nx
    normT, normAlr, normAud, normC = calc_κ(
      iy, ix, T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir=dir
    )
    κ[iy, ix] = normT * normC / (normAlr * normAud)
  end
  return κ
end

function normalize!((Clu, Cru, Cld, Crd))
  ny, nx = size(Clu)
  for iy in 1:ny, ix in 1:nx
    Clu[iy, ix] /= norm(Clu[iy, ix])
    Cld[iy, ix] /= norm(Cld[iy, ix])
    Cru[iy, ix] /= norm(Cru[iy, ix])
    Crd[iy, ix] /= norm(Crd[iy, ix])
    normC4 = scalar(Clu[iy, ix] * Cld[iy, ix] * Cru[iy, ix] * Crd[iy, ix])
    normC4 < 0 ? normClu = -abs(normC4)^(1 / 4) : normClu = normC4^(1 / 4)
    Clu[iy, ix] /= normClu
    Cld[iy, ix] /= abs(normClu)
    Cru[iy, ix] /= abs(normClu)
    Crd[iy, ix] /= abs(normClu)
  end
end

function normalize!((Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir="left")
  normalize!((Clu, Cru, Cld, Crd))
  ny, nx = size(Clu)
  for iy in 1:ny, ix in 1:nx
    Al[iy, ix] /= norm(Al[iy, ix])
    Ar[iy, ix] /= norm(Ar[iy, ix])
    Au[iy, ix] /= norm(Au[iy, ix])
    Ad[iy, ix] /= norm(Ad[iy, ix])
    iyp, ixp = per(iy + 1, ny), per(ix + 1, nx)
    normAlr = scalar(
      Clu[iy, ix] * Cru[iy, ix] * Al[iy, ix] * Ar[iy, ix] * Cld[iyp, ix] * Crd[iyp, ix]
    )
    normAlr < 0 ? normAl = -abs(normAlr)^(1 / 2) : normAl = normAlr^(1 / 2)
    Al[iy, ix] /= normAl
    Ar[iy, ix] /= abs(normAl)
    normAud = scalar(
      Clu[iy, ix] * Cld[iy, ix] * Au[iy, ix] * Ad[iy, ix] * Cru[iy, ixp] * Crd[iy, ixp]
    )
    normAud < 0 ? normAu = -abs(normAud)^(1 / 2) : normAu = normAud^(1 / 2)
    Au[iy, ix] /= normAu
    Ad[iy, ix] /= abs(normAu)
  end
end

function leftright_move!(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir="left", maxdim=5)
  ny, nx = size(T)
  P = Vector{ITensor}(undef, ny)
  P⁻ = Vector{ITensor}(undef, ny)
  if dir == "left" || dir == "up"
    xrange = 1:nx
  elseif dir == "right" || dir == "down"
    xrange = per.((nx - 1):-1:0, nx)
  end
  for ix in xrange
    ixm = per(ix - 1, nx)
    ixp = per(ix + 1, nx)
    ixpp = per(ix + 2, nx)
    for iy in 1:ny
      iym = per(iy - 1, ny)
      iyp = per(iy + 1, ny)

      Cu =
        Al[iym, ix] *
        Clu[iym, ix] *
        Au[iym, ix] *
        T[iym, ix] *
        Au[iym, ixp] *
        T[iym, ixp] *
        Cru[iym, ixpp] *
        Ar[iym, ixpp]
      @assert order(Cu) == 4
      Cd =
        Al[iy, ix] *
        Cld[iyp, ix] *
        Ad[iyp, ix] *
        T[iy, ix] *
        Ad[iyp, ixp] *
        T[iy, ixp] *
        Crd[iyp, ixpp] *
        Ar[iy, ixpp]
      @assert order(Cd) == 4
      if dir == "left" || dir == "up"
        li = commonindex(Cru[iy, ixpp], Crd[iy, ixpp])
        si = commonindex(Au[iy, ixp], Ad[iy, ixp])
      elseif dir == "right" || dir == "down"
        li = commonindex(Clu[iy, ix], Cld[iy, ix])
        si = commonindex(Au[iy, ix], Ad[iy, ix])
      end
      Cup = prime(Cu, (li, si))
      ρ = Cd * Cup
      if dir == "left" || dir == "right"
        utags = "$dir,link,x=$ixp,y=$iy"
      elseif dir == "up" || dir == "down"
        utags = "$dir,link,x=$iy,y=$ixp"
      end
      U, S, Vh, spec, u, v = svd(
        ρ, (li, si); utags=utags, vtags="tmp", maxdim=maxdim, cutoff=0.0
      )
      V = dag(Vh)
      U *= δ(u, v)
      invsqrtS = S
      for i in 1:dim(u)
        invsqrtS[i, i] = inv(sqrt(S[i, i]))
      end
      P[iy] = Cup * V * invsqrtS
      P⁻[iy] = Cd * dag(U) * invsqrtS
    end
    for iy in 1:ny
      iym = per(iy - 1, ny)
      iyp = per(iy + 1, ny)
      if dir == "left" || dir == "up"
        Al[iy, ixp] = Al[iy, ix] * P[iy] * T[iy, ix] * P⁻[iyp]
        Clu[iy, ixp] = Clu[iy, ix] * Au[iy, ix] * P⁻[iy]
        Cld[iy, ixp] = Cld[iy, ix] * Ad[iy, ix] * P[iy]
      elseif dir == "right" || dir == "down"
        Ar[iy, ixp] = Ar[iy, ixpp] * P[iy] * T[iy, ixp] * P⁻[iyp]
        Cru[iy, ixp] = Cru[iy, ixpp] * Au[iy, ixp] * P⁻[iy]
        Crd[iy, ixp] = Crd[iy, ixpp] * Ad[iy, ixp] * P[iy]
      end
    end
  end
  return normalize!((Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))
end

function swapdiag(M)
  Mp = permutedims(M, [2, 1])
  ny, nx = size(Mp)
  for iy in 1:ny, ix in 1:nx
    Mp[iy, ix] = M[ix, iy]
  end
  return Mp
end
function rotate_environment(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))
  return swapdiag(T), swapdiag.((Clu, Cld, Cru, Crd)), swapdiag.((Au, Ad, Al, Ar))
end

printdiv() = println("\n****************************************")
printstepdiv() = println("\n##################################################")

function sweepsdims(stepsizes::Vector{Int}, dims::Vector{Int})
  nstep = length(stepsizes)
  maxdims = zeros(Int, stepsizes[end])
  for i in 1:stepsizes[1]
    maxdims[i] = dims[1]
  end
  for j in 2:nstep
    for i in (stepsizes[j - 1] + 1):stepsizes[j]
      maxdims[i] = dims[j]
    end
  end
  return maxdims
end

function check_environment(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))
  ny, nx = size(T)
  for iy in 1:ny
    for ix in 1:nx
      @assert order(Clu[iy, ix]) == 2
      @assert order(Cru[iy, ix]) == 2
      @assert order(Cld[iy, ix]) == 2
      @assert order(Crd[iy, ix]) == 2
      @assert order(Al[iy, ix]) == 3
      @assert order(Ar[iy, ix]) == 3
      @assert order(Au[iy, ix]) == 3
      @assert order(Ad[iy, ix]) == 3
      @assert order(T[iy, ix]) == 4
      @assert order(T[iy, ix]) == 4
      @assert order(T[iy, ix]) == 4
      @assert order(T[iy, ix]) == 4
      @assert length(commoninds(Clu[iy, ix], Cru[iy, ix])) == 1
      @assert length(commoninds(Clu[iy, ix], Cld[iy, ix])) == 1
      @assert length(commoninds(Cld[iy, ix], Crd[iy, ix])) == 1
      @assert length(commoninds(Cru[iy, ix], Crd[iy, ix])) == 1
    end
  end
end

function ctmrg(T::Matrix{ITensor}, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); verbose=false)
  ny, nx = size(T)

  verbose && println("Original:")
  verbose && @show calc_κ(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))

  nstep = 1000
  maxdim = 10
  dirs = ["left", "up", "right", "down"]
  for ctmrg_step in 1:nstep
    verbose && printstepdiv()
    dir = dirs[per(ctmrg_step, length(dirs))]
    verbose && @show ctmrg_step, dir

    if dir == "left" || dir == "right"
      leftright_move!(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir=dir, maxdim=maxdim)
    elseif dir == "up" || dir == "down"
      T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad) = rotate_environment(
        T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad)
      )
      leftright_move!(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir=dir, maxdim=maxdim)
      T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad) = rotate_environment(
        T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad)
      )
    end

    check_environment(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad))

    verbose && @show Mκ = calc_κ(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir=dir)
    verbose && @show abs(prod(vec(Mκ)))^(1 / (nx * ny))
  end
  Mκ = calc_κ(T, (Clu, Cru, Cld, Crd), (Al, Ar, Au, Ad); dir=dir)
  κave = abs(prod(vec(Mκ)))^(1 / (nx * ny))
  return κave
end
