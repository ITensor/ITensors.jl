N = 20
Npart = 10
t1 = 1.0
t2 = 0.2
U = 1.0
V1 = 0.5
Nsweep = 6
sweeps_args =
  [
   "maxdim" "mindim" "cutoff" "noise"
     50      10       1e-12    1e-7
    100      20       1e-12    1e-8
    200      20       1e-12    1e-10
    400      20       1e-12    0
    800      20       1e-12    1e-11
    800      20       1e-12    0
  ]

using ITensors

#
# New ITensor functionality
#

#get_maxdims(sw::Sweeps) = [maxdim(sw, n) for n in 1:nsweep(sw)]
#get_mindims(sw::Sweeps) = [mindim(sw, n) for n in 1:nsweep(sw)]
#get_cutoffs(sw::Sweeps) = [cutoff(sw, n) for n in 1:nsweep(sw)]
#get_noises(sw::Sweeps) = [noise(sw, n) for n in 1:nsweep(sw)]
#
#parse_args(; first_arg::Int = 1) = parse_args(ARGS; first_arg = first_arg)

#function parse_args(args_list::Vector; first_arg::Int = 1,
#                                       delimiter = '=')
#  parsed = Dict{Symbol, Any}()
#  for n in first_arg:length(args_list)
#    a = args_list[n]
#    opt, arg = split(a, delimiter)
#    parsed[Symbol(opt)] = arg
#  end
#  return parsed
#end

# Create the sweeps object
sweeps = Sweeps(Nsweep, sweeps_args)
# Extract the original (default) values
Maxdim = get_maxdims(sweeps)
Mindim = get_mindims(sweeps)
Cutoff = get_cutoffs(sweeps)
Noise = get_noises(sweeps)

# Parse arguments to overide defaults
args = parse_args(; first_arg = 2)
# Evaluate the arguments as variables
for (arg, val) in args
  eval(Meta.parse("$arg = $val"))
end

