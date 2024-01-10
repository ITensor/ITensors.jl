
default_compile_dir() = joinpath(homedir(), ".julia", "sysimages")

default_compile_filename() = "sys_itensors.so"

default_compile_path() = joinpath(default_compile_dir(), default_compile_filename())

# SOURCE: https://github.com/JuliaLang/PackageCompiler.jl/blob/adbf1df36aadc245f95d1282dda02d8bbeccb32f/src/PackageCompiler.jl#L166
function get_julia_cmd()
  julia_path = joinpath(Sys.BINDIR, Base.julia_exename())
  color = if Base.have_color === nothing
    "auto"
  elseif Base.have_color
    "yes"
  else
    "no"
  end
  return `$julia_path --color=$color --startup-file=no`
end

function compile_note(; dir=default_compile_dir(), filename=default_compile_filename())
  path = joinpath(dir, filename)
  return """
  You will be able to start Julia with a compiled version of ITensors using:

  ```
  ~ julia --sysimage $path
  ```

  and you should see that the startup times and JIT compilation times are substantially improved when you are using ITensors.

  In unix, you can create an alias with the Bash command:

  ```
  ~ alias julia_itensors="julia --sysimage $path -e 'using ITensors' -i"
  ```

  which you can put in your `~/.bashrc`, `~/.zshrc`, etc. This also executes
  `using ITensors` so that ITensors is loaded and ready to use, you can leave off `
  -e 'using ITensors' -i` if you don't want that. Then you can start Julia with a
  version of ITensors installed with the command:

  ```
  ~ julia_itensors
  ```

  Note that if you update ITensors to a new version, for example with `using
  Pkg; Pkg.update("ITensors")`, you will need to run the `ITensors.compile()`
  command again to recompile the new version of ITensors.
  """
end

function compile(;
  dir::AbstractString=default_compile_dir(),
  filename::AbstractString=default_compile_filename(),
  kwargs...,
)
  if !isdir(dir)
    println("""The directory "$dir" doesn't exist yet, creating it now.""")
    println()
    mkdir(dir)
  end
  path = joinpath(dir, filename)
  println(
    """Creating the system image "$path" containing the compiled version of ITensors. This may take a few minutes.""",
  )

  include_MKL::Bool = get(kwargs, :include_MKL, false)
  num_threads::Int = get(kwargs, :num_threads, 1)
  blocksparse_multithreading::Bool = get(kwargs, :blocksparse_multithreading, false)
  contraction_sequence_optimization::Bool = get(
    kwargs, :contraction_sequence_optimization, true
  )

  project = dirname(Base.active_project())

  # define tracefile that contains the objects the PackageCompiler has to compile
  # tracefile = joinpath(@__DIR__, "tracefile.jl")
  tracefile, io_tracefile = mktemp(; cleanup=false)
  close(io_tracefile)

  # Define script from which a julia process will watch which objects are to be compiled in a new system image
  # key feature here is that we control the process itself that generates the tracefile,
  # which allows us to let the process be multithreaded (or not)
  # precompile_file = joinpath(@__DIR__, "script_to_watch.jl")
  precompile_file, io_precompile_file = mktemp(; cleanup=false)

  # write the dynamically generated precompilation file to disk
  # reflecting the user choice of multithreading enabled/disabled and or MKL, etc.
  # First convert Expr object of script_to_watch() to a string that can be written to disk

  write(
    io_precompile_file,
    string(
      script_to_watch(;
        include_MKL=include_MKL,
        num_threads=num_threads,
        blocksparse_multithreading=blocksparse_multithreading,
        contraction_sequence_optimization=contraction_sequence_optimization,
      ),
    ),
  )
  close(io_precompile_file)

  # Cruical advantage here: controlling the process that produces the tracefile and possibly enable multithreading
  # adaption from https://github.com/JuliaLang/PackageCompiler.jl/ 
  if num_threads > 1
    precompile_cmd = `$(get_julia_cmd()) -t $(num_threads) --compile=all --trace-compile=$tracefile $(precompile_file)`
  else
    precompile_cmd = `$(get_julia_cmd()) --compile=all --trace-compile=$tracefile $(precompile_file)`
  end

  # change build arguments of sysimage accordingly
  build_args = get(kwargs, :build_args, `-t $(num_threads)`)

  # add splitted paths to PATH depending on system
  splitter = Sys.iswindows() ? ';' : ':'
  # add environment variables to execution command
  precompile_cmd = addenv(precompile_cmd, "JULIA_LOAD_PATH" => "$project$(splitter)@stdlib")
  # run the julia process that generates the tracefile,
  # which will be used to compile the system image.
  run(precompile_cmd)

  package_list = include_MKL == true ? [:ITensors, :MKL] : [:ITensors]

  create_sysimage(
    package_list;
    sysimage_path=path,
    # precompile_execution_file=joinpath(@__DIR__, "precompile_itensors.jl"),
    precompile_statements_file=tracefile,
    sysimage_build_args=build_args,
  )
  println(compile_note(; dir=dir, filename=filename))
  return path
end

@doc """
    ITensors.compile(; dir = "$(default_compile_dir())",
                       filename = "$(default_compile_filename())",
                       build_args::Cmd=`-t <num_threads>`,
                       include_MKL::Bool = false,
                       num_threads::Int = 1,
                       blocksparse_multithreading::Bool = false,
                       contraction_sequence_optimization::Bool = true
                       )

Compile ITensors.jl with [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/).
This will take some time, perhaps a few minutes.

This will create a system image containing the compiled version of ITensors
located at `dir/filename`, by default `$(default_compile_path())`.

 - `build_args::AbstractString`: A set of command line options that is used in the Julia process building the sysimage, for example `"-O1"`. Note that if you specify it explicitely and want to use multithreading make sure to include `-t <num_threads>`.
 
 - `num_threads::Int` Sets the number of threads the functions in the system image are using. If `blocksparse_multithreading=false` then the set amount of threads are allocated to `BLAS.set_num_threads(num_threads)`, cf. below. Note ITensors.Strided multithreading is disabled.

 - `include_MKL::Bool` Decide to include MKL.jl in the system image (`include_MKL=true`) or not (`include_MKL=false`).

 - `blocksparse_multithreading::Bool` decide to use the multithreading features of BlockSparse ITensors when conserving QNs.
    Note that this requires `num_threads>1`. If set `false` when `num_threads>1` then `BLAS.set_num_threads(num_threads)`. If set `true`, the blocksparse_multithreading will be enabled. Note Strided multithreading is disabled.

 - `contraction_sequence_optimization::Bool` use `ITensors.enable_contraction_sequence_optimization()` in system image

$(compile_note())
""" compile

# this function is needed as we dynamically need to adapt the execution file
# to reflect the multithreading features (or not)
# script_to_watch() returns an `Expr` object
function script_to_watch(;
  include_MKL::Bool=false,
  num_threads::Int=1,
  blocksparse_multithreading::Bool=false,
  contraction_sequence_optimization::Bool=true,
)
  quote
    using ITensors, LinearAlgebra
    $(include_MKL) == true && using MKL

    # check if num_threads is a positive integer
    if $(num_threads) <= 0
      throw(ArgumentError("num_threads must be positive integer but is $(num_threads)!"))
    end

    # consider two cases for multithreading:
    # 1. BlockSparse multithreading with QN conservation
    if $(num_threads) > 1 && $(blocksparse_multithreading)
      BLAS.set_num_threads(1)
      ITensors.Strided.set_num_threads(1)
      ITensors.enable_threaded_blocksparse()
      # 2. BLAS multithreading
    elseif $(num_threads) > 1 && !$(blocksparse_multithreading)
      BLAS.set_num_threads(num_threads)
      ITensors.Strided.set_num_threads(1)
      ITensors.disable_threaded_blocksparse()
    end

    $(contraction_sequence_optimization) &&
      ITensors.enable_contraction_sequence_optimization()

    # what follows is the basis script to be watched for compilation
    N = 6
    sweeps = Sweeps(3)
    maxdim!(sweeps, 10)
    cutoff!(sweeps, 1E-13)

    ampo = OpSum()
    for j in 1:(N - 1)
      ampo .+= "Sz", j, "Sz", j + 1
      ampo .+= 0.5, "S+", j, "S-", j + 1
      ampo .+= 0.5, "S-", j, "S+", j + 1
    end

    sites = siteinds("S=1", N)
    H = MPO(ampo, sites)
    psi0 = randomMPS(sites; linkdims=2)
    dmrg(H, psi0, sweeps)

    sites_qn = siteinds("S=1", N; conserve_qns=true)
    if !hasqns(sites_qn[1])
      throw(ErrorException("Index does not have QNs in part of precompile script"))
    end
    H_qn = MPO(ampo, sites_qn)
    psi0_qn = randomMPS(sites_qn, [isodd(n) ? "Up" : "Dn" for n in 1:N]; linkdims=2)
    dmrg(H_qn, psi0_qn, sweeps)
  end
end
