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

function volumetric_liquid_fraction(ϑ_l::FT, ν_eff::FT) where {FT}
    if ϑ_l < ν_eff
        θ_l = ϑ_l
    else
        θ_l = ν_eff
    end
    return θ_l
end

function matric_potential(α::FT, n::FT, m::FT, S::FT) where {FT}
    ψ_m = -((S^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
    return ψ_m
end

function effective_saturation(porosity::FT, ϑ_l::FT, θr::FT) where {FT}
    ϑ_l_safe = max(ϑ_l, θr + eps(FT))
    S_l = (ϑ_l_safe - θr) / (porosity - θr)
    return S_l
end

function pressure_head(
    α::FT,
    n::FT,
    m::FT,
    θ_r::FT,
    ϑ_l::FT,
    ν_eff::FT,
    S_s::FT,
) where {FT}
    S_l_eff = effective_saturation(ν_eff, ϑ_l, θ_r)
    if S_l_eff <= FT(1.0)
        ψ = matric_potential(α,n,m, S_l_eff)
    else
        ψ = (ϑ_l - ν_eff) / S_s
    end
    return ψ
end

function hydraulic_conductivity(
    Ksat::FT,
    m::FT,
    S::FT
) where {FT}
    if S < FT(1)
        K = sqrt(S) * (FT(1) - (FT(1) - S^(FT(1) / m))^m)^FT(2)
    else
        K = FT(1)
    end
    return K* Ksat
end

const ft = Float64;

ν = ft(0.495);
Ksat = ft(0.0443 / 3600 / 100); # m/s
S_s = ft(1e-3); #inverse meters
vg_n = ft(2.0);
vg_α = ft(2.6); # inverse meters
vg_m = ft(1) - ft(1) / vg_n;
θ_r = ft(0);

zmax = ft(0);
zmin = ft(-10);
const n = 50;
const Δz = (zmax-zmin)/n
const z = Array((zmin+Δz/ft(2.0)):Δz:(zmax-Δz/ft(2.0)))

top_flux_bc = ft(0.0);
bot_flux_bc = ft(0.0);

p = [ν,vg_α, vg_n, vg_m, Ksat, S_s, θ_r, top_flux_bc, bot_flux_bc];

t0 = ft(0);
tf = ft(60 * 60 * 24 * 36);
dt = ft(100);
θ0 =  ft(0.494) .+ zeros(n)
K = zeros(n)
ψ = zeros(n)
F = zeros(n)
function rhs_flux!(dY, Y, p, t)
    ν, vg_α, vg_n, vg_m, Ksat, S_s, θ_r, top_flux_bc, bot_flux_bc = p
    @. K =  hydraulic_conductivity(Ksat,vg_m,
                                       effective_saturation(ν, Y, θ_r)
                                   )
    @. ψ = pressure_head(vg_α,vg_n,vg_m, θ_r,Y, ν, S_s) + z

    @inbounds for i in 1:1:50
        ip1, im1 = i+1, i-1
        Fi_ph = ip1 == n+1 ? ft(top_flux_bc) : ft(-2.0)/(ft(1.0)/K[ip1]+ft(1.0)/K[i])*(ψ[ip1]-ψ[i])/Δz
        Fi_mh = im1  == 0 ? ft(bot_flux_bc) : ft(-2.0)/(ft(1.0)/K[i]+ft(1.0)/K[im1])*(ψ[i]-ψ[im1])/Δz
        dY[i] = -ft(1.0)/Δz*(Fi_ph-Fi_mh)
    end

    
end

prob = ODEProblem(rhs_flux!, θ0, (t0, tf), p);
input = rand(n);
output = similar(input);
# this is cool, but it's just tridiagnol -dont need it?
sparsity_pattern = jacobian_sparsity(rhs_flux!, output, input, p, 0.0);
jac_sparsity = Float64.(sparse(sparsity_pattern));
f = ODEFunction(rhs_flux!;jac_prototype=jac_sparsity);
prob_sparse = ODEProblem(f, θ0, (t0,tf), p);

### Fixing the timestep for all algorithms
println("Default")
@btime solve(prob, dt= dt, save_every_step =false, adaptive = false);
println("Rosenbrock23 - finite diff Jacobian")
@btime solve(prob, Rosenbrock23(autodiff=false,diff_type=Val{:central}), dt= dt, save_every_step = false, adaptive = false);
println("Implicit Euler, FD")
@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= dt, save_every_step = false, adaptive = false);
println("Default, sparse")
@btime solve(prob_sparse, dt= dt, save_every_step = false, adaptive = false);
println("Rosenbrock FD, sparse")
@btime solve(prob_sparse, Rosenbrock23(autodiff=false,diff_type=Val{:central}), dt= dt, save_every_step = false, adaptive = false);
println("ImplicitEuler FD, sparse")
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= dt, save_every_step = false, adaptive = false);
println("KenCarp4 FD, sparse")
@btime solve(prob_sparse, KenCarp4(autodiff=false,diff_type=Val{:central}), dt= dt, save_every_step = false, adaptive = false);
println("SSPRK33")
@btime solve(prob, SSPRK33(), dt = dt, save_every_step = false, adaptive = false);
println("Forward Euler")
@btime solve(prob, Euler(), dt = dt, save_every_step = false, adaptive = false);


#de = modelingtoolkitize(prob)
#jac = eval(ModelingToolkit.generate_jacobian(de)[2])
#f = ODEFunction(rhs_flux!, jac=jac)
#prob_jac = ODEProblem(f, θ0, (t0, tf), p)
