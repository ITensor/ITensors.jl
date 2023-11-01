## I don't think we need this function anymore
# function NDTensors.svd_catch_error(A::CuMatrix; alg="JacobiAlgorithm")
#   if alg == "JacobiAlgorithm"
#     alg = CUDA.CUSOLVER.JacobiAlgorithm()
#   else
#     alg = CUDA.CUSOLVER.QRAlgorithm()
#   end
#   USV = try
#     svd(expose(A); alg=alg)
#   catch
#     return nothing
#   end
#   return USV
# end
