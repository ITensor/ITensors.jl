
# Timing and Profiling your code

It is very important to time and profile your code to make sure your code is running as fast as possible. Here are some tips on timing and profiling your code.

If you are concerned about the performance of your code, a good place to start is Julia's [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/).

## Timing and benchmarking

Julia has many nice timing tools available. Tools like [@time](https://docs.julialang.org/en/v1/base/base/#Base.@time) can be used to measure the time of specific lines of code. For microbenchmarking, we recommend the [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) package.

## ProfileView.jl

Julia has a built-in profiler in the standard module [Profile](https://docs.julialang.org/en/v1/manual/profile/). You can use it as follows:
```julia
julis> using ITensors

julia> using Profile

julia> @profile include(joinpath(pkgdir(ITensors), "examples", "dmrg", "1d_heisenberg_conserve_spin.jl"))
After sweep 1 energy=-137.995732867390 maxlinkdim=10 maxerr=1.93E-02 time=0.862
After sweep 2 energy=-138.801057557054 maxlinkdim=20 maxerr=3.37E-05 time=1.126
After sweep 3 energy=-138.940075984826 maxlinkdim=91 maxerr=9.99E-11 time=1.880
After sweep 4 energy=-138.940086063995 maxlinkdim=99 maxerr=1.00E-10 time=3.033
After sweep 5 energy=-138.940086076330 maxlinkdim=95 maxerr=9.97E-11 time=2.824
Final energy = -138.940086076330

julia> Profile.print()
Overhead ╎ [+additional indent] Count File:Line; Function
=========================================================
   ╎6576 @Base/client.jl:485; _start()
   ╎ 6576 @Base/client.jl:302; exec_options(opts::Base.JLOptions)
   ╎  6576 @Base/client.jl:372; run_main_repl(interactive::Bool, quiet::Bool, b...
   ╎   6576 @Base/essentials.jl:706; invokelatest
   ╎    6576 @Base/essentials.jl:708; #invokelatest#2
   ╎     6576 @Base/client.jl:387; (::Base.var"#874#876"{Bool, Bool, Bool})(REPL...
   ╎    ╎ 6576 .../stdlib/v1.6/REPL/src/REPL.jl:305; run_repl(repl::REPL.AbstractREPL, consumer::Any)
   ╎    ╎  6576 ...stdlib/v1.6/REPL/src/REPL.jl:317; run_repl(repl::REPL.AbstractREPL, consumer::...
[...]
```
We don't actually recommend printing it since the output is extremely long and difficult to read. Instead, it is best to use a visualization tool to visualize the profile, so you can see at a glance what parts of your code are taking up the most time.

For a quick and easy way of seeing how fast different parts of your code are, we highly recommend [ProfileView.jl](https://github.com/timholy/ProfileView.jl). All you have to do is load `ProfileView` and append a simple macro `@profview`:
```julia
julia> using ITensors

julia> using ProfileView

julia> @profview include(joinpath(pkgdir(ITensors), "examples", "dmrg", "1d_heisenberg_conserve_spin.jl"));
After sweep 1 energy=-137.995732867390 maxlinkdim=10 maxerr=1.93E-02 time=0.977
After sweep 2 energy=-138.801057557054 maxlinkdim=20 maxerr=3.37E-05 time=1.252
After sweep 3 energy=-138.940075984826 maxlinkdim=91 maxerr=9.99E-11 time=2.263
After sweep 4 energy=-138.940086063995 maxlinkdim=99 maxerr=1.00E-10 time=2.938
After sweep 5 energy=-138.940086076330 maxlinkdim=95 maxerr=9.97E-11 time=2.988
Final energy = -138.940086076330
``` 
A window will pop up with a "flame graph", where the width of a bar corresponds to the percentage of time the function call took, and as you go up in the graph you see timings for more and more nested functions. The graph will still look quite complicated for this case, but at larger bond dimensions you should see that certain functions like matrix multiplications and decompositions will start to dominate.

## TimerOutputs.jl

For more focused timing of specific parts of your code, we recommend the package [TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl). This can help measure lines of code that are called repeatedly, where you want to know the accumulated time. We have some timers defined internally in ITensors.jl for the `dmrg` function which can be called as follows:
```julia
julia> using ITensors

julia> ITensors.TimerOutputs.enable_debug_timings(ITensors)
timeit_debug_enabled (generic function with 1 method)

julia> ITensors.TimerOutputs.reset_timer!(ITensors.NDTensors.timer)
 ──────────────────────────────────────────────────────────────────
                           Time                   Allocations      
                   ──────────────────────   ───────────────────────
 Tot / % measured:     59.4μs / 0.00%              992B / 0.00%    

 Section   ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────
 ──────────────────────────────────────────────────────────────────

julia> include("1d_heisenberg_conserve_spin.jl")
After sweep 1 energy=-137.995732867390 maxlinkdim=10 maxerr=1.93E-02 time=0.597
After sweep 2 energy=-138.801057557054 maxlinkdim=20 maxerr=3.37E-05 time=0.798
After sweep 3 energy=-138.940075984826 maxlinkdim=91 maxerr=9.99E-11 time=1.285
After sweep 4 energy=-138.940086063995 maxlinkdim=99 maxerr=1.00E-10 time=1.878
After sweep 5 energy=-138.940086076330 maxlinkdim=95 maxerr=9.97E-11 time=1.936
Final energy = -138.940086076330

julia> ITensors.NDTensors.timer
 ────────────────────────────────────────────────────────────────────────────────
                                         Time                   Allocations      
                                 ──────────────────────   ───────────────────────
        Tot / % measured:             14.4s / 45.0%           9.59GiB / 95.1%    

 Section                 ncalls     time   %tot     avg     alloc   %tot      avg
 ────────────────────────────────────────────────────────────────────────────────
 dmrg: eigsolve             990    4.72s  72.7%  4.76ms   7.27GiB  79.7%  7.52MiB
 dmrg: replacebond!         990    1.31s  20.2%  1.32ms   1.12GiB  12.2%  1.15MiB
 dmrg: position!            990    360ms  5.55%   364μs    571MiB  6.11%   590KiB
 dmrg: psi[b]*psi[b+1]      990    106ms  1.63%   107μs    182MiB  1.95%   188KiB
 ────────────────────────────────────────────────────────────────────────────────
```
