using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
  "--N", "-N"
    help = "Number of sweeps"
    arg_type = Int
    default = 20
  "--Npart"
    help = "Number of particles"
    arg_type = Int
    default = 10
  "--t1"
    help = "Nearest neighbor hopping"
    arg_type = Float64
    default = 1.0
  "--t2"
    help = "Next-nearest neighbor hopping"
    arg_type = Float64
    default = 0.2
  "--U", "-U"
    help = "On-site potential"
    arg_type = Float64
    default = 1.0
  "--V1", "-V"
    help = "Nearest neighbor potential"
    arg_type = Float64
    default = 0.5
  "--nsweep", "-n"
    help = "Number of sweeps of DMRG"
    arg_type = Int64
    default = 6
  "--maxdim"
    help = "Maximum bond dimension in DMRG"
    nargs = '+'
    arg_type = Int64
    default = [50, 100, 200, 400, 800]
  "--mindim"
    help = "Minimum bond dimension in DMRG"
    nargs = '+'
    arg_type = Int64
    default = [10, 20]
  "--cutoff", "-c"
    help = "Truncation cutoff in DMRG"
    nargs = '+'
    arg_type = Float64
    default = [1e-12]
  "--noise"
    help = "Magnitude of noise to add in DMRG"
    nargs = '+'
    arg_type = Float64
    default = [1e-7, 1e-8, 1e-10, 0, 1e-11, 0]
  "arg1"
    help = "Input file"
    arg_type = String
end
# Parse the arguments and define
# variables for all of the inputs
parsed_args = parse_args(s)
for (arg, val) in parsed_args
  arg == "arg1" && continue
  eval(Meta.parse("$arg = $val"))
end
