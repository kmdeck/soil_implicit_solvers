include("Domains.jl")
using .Domains: make_function_space, Column
include("Richards.jl")
using .Richards: rhs_flux!
using ClimaCore
using DifferentialEquations
using OrdinaryDiffEq:
    ODEProblem,
    solve,
    Rosenbrock23,
    KenCarp4,
    ImplicitEuler
using ModelingToolkit
using BenchmarkTools
using SparsityDetection, SparseArrays
abstract type AbstractParams end
struct Parameters{f <:AbstractFloat} <:AbstractParams 
    ν::f
    vg_α::f
    vg_n::f
    vg_m::f
    Ksat::f
    S_s::f
    θ_r::f
    top_flux_bc::f
    bot_flux_bc::f
    z::ClimaCore.Fields.Field
end

FT = Float64;

ν = FT(0.495);
Ksat = FT(0.0443 / 3600 / 100); # m/s
S_s = FT(1e-3); #inverse meters
vg_n = FT(2.0);
vg_α = FT(2.6); # inverse meters
vg_m = FT(1) - FT(1) / vg_n;
θ_r = FT(0);

zmax = FT(0);
zmin = FT(-10);
n = 50;
domain = Column(FT, zlim = (zmin, zmax), nelements = n);
cs, = make_function_space(domain);
z =  getproperty(ClimaCore.Fields.coordinate_field(cs), :z);

top_flux_bc = FT(0.0);
bot_flux_bc = FT(0.0);

p = Parameters{FT}(ν,vg_α, vg_n, vg_m, Ksat, S_s, θ_r, top_flux_bc, bot_flux_bc, z);

t0 = FT(0);
tf = FT(60 * 60 * 24 * 36);
dt = FT(100);
θ0 = ClimaCore.Fields.FieldVector(;ϑ_l = ClimaCore.Fields.zeros(cs) .+ FT(0.494))
#θ0 = ClimaCore.Fields.zeros(cs) .+ FT(0.494);


prob = ODEProblem(rhs_flux!, θ0, (t0, tf), p);

@btime solve(prob);
#@btime solve(prob, Rosenbrock23())
@btime solve(prob, Rosenbrock23(autodiff=false,diff_type=Val{:central}));


input = rand(n,n,1)
output = similar(input)
sparsity_pattern = jacobian_sparsity(rhs_flux!, output, input, p, 0.0)
#de = modelingtoolkitize(prob)
#jac = eval(ModelingToolkit.generate_jacobian(de)[2])
#f = ODEFunction(rhs_flux!, jac=jac)
#prob_jac = ODEProblem(f, θ0, (t0, tf), p)
