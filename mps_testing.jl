using Pkg
Pkg.develop("ITensors")


using LinearAlgebra
using ITensors,Test, TimerOutputs, Profile, TupleTools,PermutedArrays, BenchmarkTools, ProfileView

using MKL_jll, AppleAccelerate, AppleAccelerateLinAlgWrapper, MKL, BLISBLAS
#BLAS.get_config()
#BLAS.lbt_find_backing_library("dgemm_", :lp64)
#BLAS.set_num_threads(10)
#const tmr = TimerOutput();

function optimize_kernel(x_in::ITensor, tn_map::Vector{ITensor}, iters::Int, buffer = false)
  timer = TimerOutput()
  x_out = x_in
  if buffer
    
    bufsize = ITensors.compute_buffer_size([x_in; tn_map]) 
    for i in [x_in; tn_map]
      size = dim(i)
      bufsize = (size > bufsize ? size : bufsize)
    end
    #bufsize = bufsize + Int(bufsize * 0.5)
    println("buffer size: ", bufsize * 8 / 1e9, " Gb")
    b = Vector{eltype(storage(x_in))}(undef, bufsize)
    ba = Vector{eltype(storage(x_in))}(undef, bufsize)
    bb = Vector{eltype(storage(x_in))}(undef, bufsize)
    @timeit timer "buffer" begin
      for _ in 1:iters
        x_out = contract([x_out; tn_map];buf=b,timer=timer,buf_a=ba, buf_b=bb)
        noprime!(x_out)
        normalize!(x_out)
      end
    end
  else
    @timeit timer "no buffer" begin
        x_out = x_in
        for _ in    1:iters
          x_out = contract([x_out; tn_map]; timer=timer)
          noprime!(x_out)
          normalize!(x_out)
        end
      end
  end
  println(timer)
  return x_out
end

chi = 1000 # MPS dimension
d = 2 # local dimension
D = 50 # MPO dimension
iters = 10 # Number of times to perform the TN contraction

l, r, s1, s2, h1, h2, h3 = Index.((chi, chi, d, d, D, D, D))
psi = randomITensor(l, s1, s2, r)
L = randomITensor(h1,l, l')
H1 = randomITensor(h1, s1, s1', h2)
H2 = randomITensor(h2, s2, s2', h3)
R = randomITensor(r, h3, r')

phi = optimize_kernel(psi, [L, H1, H2, R], iters)
phi = optimize_kernel(psi, [L, H1, H2, R], iters, true)

@btime ITensors.contract!(C, A, B)
let 
  BLAS.set_num_threads(10)
  timer = TimerOutput()
  for ntimes in 1:100
    for i in [10000]
      edge1 = Index(i);
      edge2 = Index(i);
      for j in [10000]
        mid = Index(j);
        A = randomITensor(edge1, mid);
        B = randomITensor(mid, edge2);
        C = randomITensor(edge1, edge2);
        @timeit timer "hello" ITensors.contract(C, A, B)
      end
    end
  end
  show(timer,allocations = false, sortby=:firstexec)
end

timer = TimerOutput()
let 
  timer = TimerOutput()
  for ntimes in 1:20
    for i in [10, 50, 100, 500, 1000, 5000]
      edge1 = Index(i);
      edge2 = Index(i);
      for j in [10, 50, 100, 500, 1000, 5000]
        mid = Index(j);
        A = randomITensor(edge1, mid);
        B = randomITensor(mid, edge2);
        str = "$i / $j / $i"
        @timeit timer str C = A * B
      end
    end
  end
  show(timer,allocations = false, sortby=:firstexec)
end

# ##
  # push!(tensor_array, randomITensor(j,k,m));
  # push!(tensor_array, randomITensor(k,l));

  # pA = NTuple{3,Int}((2,1,3))A = Base.ReshapedArray(NDTensors.data(storage(tensor_array[1])), dims(inds(tensor_array[1])), ());
  # permutedims(A, pA)

  # C = ITensor(undef, noncommoninds(tensor_array[2], tensor_array[1]))
  # println(dim(C) * 8 / 1e9)
  # println("Giga ops: ", 100 * dim(i) * dim(r) * dim(j) * dim(m) * dim(k) / 1e9)
  # ba = Vector{Float64}(undef, dim(tensor_array[1]));
  # @show inds(tensor_array[1])
  # @show inds(tensor_array[2])
  # @show inds(C)
  # ITensors.contract!(C, tensor_array[1], tensor_array[2];timer=timer)
  # println(timer)
  # i = Index(200, "i")
  # j = Index(30, "j")
  # k = Index(100, "k")
  # l = Index(38, "l")
  # m = Index(340, "m")
  # n = Index(87, "n")
  # r = Index(20, "r")
  # tensor_array = [randomITensor(r,i,j,m)];
  # push!(tensor_array, randomITensor(n,i,));
  # C = ITensor(0,noncommoninds(tensor_array[1], tensor_array[2]))

  # begin
  #   timer = TimerOutput();
  # for i = 1:5
  #   ITensors.contract!(C, tensor_array[1], tensor_array[2];timer=timer)
  #   #contract(tensor_array[1], tensor_array[2])
  # end
  #   println(timer)
  # end
  # function test(A, B)
  #   #push!(tensor_array, randomITensor(l,m,n));
  #   #seq = ITensors.optimal_contraction_sequence(tensor_array)

  #   tensor_array = [A,B]
  #   bsize = ITensors.compute_buffer_size(tensor_array, [1,2])
  #   d = NDTensors.Dense(1)
  #   is = IndexSet(noncommoninds(B,A))
  #   C = ITensor(is, d)

  #   # ba = Vector{Float64}(undef, size(storage(tensor_array[1])))
  #   # bb = Vector{Float64}(undef, size(storage(tensor_array[2])))
  #   # fill!(ba, 0); fill!(bb, 0)
  #   ba = Vector{Float64}(undef,dim(A))
  #   @show length(ba)
  #   #ITensors.contract!(C,A,B, 1.0, 0.0; buf_a = ba)
  #   #C = ITensor(is, d)
  #   ITensors.contract!(C,A,B, 1.0, 0.0)

  #   #f1 = ITensors.contract(tensor_array, sequence="left_associative")
  # end

  # test(tensor_array[1], tensor_array[2])

  # # Profile.clear_malloc_data()

  # # test(tensor_array[1], tensor_array[2])
# ##