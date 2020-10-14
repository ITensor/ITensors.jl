using ITensors

function ctmrg(T::ITensor,
               Clu::ITensor,
               Al::ITensor;
               χmax::Int, nsteps::Int)
  Clu = addtags(Clu, "orig", "link")
  Al = addtags(Al, "orig", "link")
  for i in 1:nsteps
    ## Get the grown corner transfer matrix (CTM)
    Au = replacetags(Al, "down,link", "left,link")
    Au = replacetags(Au, "up,link", "right,link")
    Au = replacetags(Au, "left,site", "up,site")

    Clu⁽¹⁾ = Clu * Al * Au * T

    ## Diagonalize the grown CTM
    ld = firstind(Clu⁽¹⁾, "link,down")
    sd = firstind(Clu⁽¹⁾, "site,down")
    lr = firstind(Clu⁽¹⁾, "link,right")
    sr = firstind(Clu⁽¹⁾, "site,right")

    Cdr, Ur = eigen(Clu⁽¹⁾, (ld, sd), (lr, sr);
                    ishermitian = true,
                    maxdim = χmax,
                    lefttags = "link,down,renorm",
                    righttags = "link,right,renorm")

    ## The renormalized CTM is the diagonal matrix of eigenvalues
    Clu = replacetags(Cdr, "renorm", "orig")
    Clu = replacetags(Clu, "down", "up")
    Clu = replacetags(Clu, "right", "left")
    Clu = Clu / norm(Clu)

    ## Calculate the renormalized half row transfer matrix (HRTM)
    Ud = replacetags(Ur, "right", "down")
    Uu = replacetags(Ud, "down", "up")

    Al = Al * Uu * T * Ud
    Al = replacetags(Al, "renorm", "orig")
    Al = replacetags(Al, "right,site", "left,site")
    Al = Al / norm(Al)
  end
  Clu = removetags(Clu, "orig")
  Al = removetags(Al, "orig")
  return Clu, Al
end

