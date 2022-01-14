using DifferentialEquations
using OrdinaryDiffEq:
    ODEProblem,
    solve,
    ImplicitEuler,
    ImplicitMidpoint,
    Midpoint,
    Euler,
    SSPRK33
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
    S_l::FT = (ϑ_l_safe - θr) / (porosity - θr)
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
vg_n = ft(1.4);
vg_α = ft(2.6); # inverse meters
vg_m = ft(1) - ft(1) / vg_n;
θ_r = ft(0.124);

zmax = ft(0);
zmin = ft(-1.5);
const n = 150;
const Δz = (zmax-zmin)/n
const z = Array((zmin+Δz/ft(2.0)):Δz:(zmax-Δz/ft(2.0)))


p = [ν,vg_α, vg_n, vg_m, Ksat, S_s, θ_r];

t0 = ft(0);
tf = ft(60*60*24);
θ0 =  ft(0.24) .+ zeros(n)

function rhs_flux!(dY, Y, p, t)
    ν,
    vg_α,
    vg_n,
    vg_m,
    Ksat,
    S_s,
    θ_r = p
    K =  hydraulic_conductivity.(Ksat,vg_m,
                                 effective_saturation.(ν, Y, θ_r))
    ψ = pressure_head.(vg_α,vg_n,vg_m, θ_r,Y, ν, S_s) .+ z
    
    top = -0.5*(Ksat+K[end])*(0.0-ψ[end])/Δz
    @inbounds for i in 1:1:n
        ip1, im1 = i+1, i-1
        Fi_ph::ft = ip1 == n+1 ? ft.(top) : ft(-2.0)/(ft(1.0)/K[ip1]+ft(1.0)/K[i])*(ψ[ip1]-ψ[i])/Δz
        Fi_mh::ft = im1  == 0 ? ft.(-K[1]) : ft(-2.0)/(ft(1.0)/K[i]+ft(1.0)/K[im1])*(ψ[i]-ψ[im1])/Δz
        dY[i] = -ft(1.0)/Δz*(Fi_ph-Fi_mh)
    end

    
end

prob = ODEProblem(rhs_flux!, θ0, (t0, tf), p);
## Which timestep we can use for same level of error
truth = solve(prob, dt = 1.0,SSPRK33(),save_every_step =false)
u = truth.u[end]


i = Array(1:1:n)
iu = Array(1:1:n-1)
il = Array(2:1:n)
i1 = vcat(i,iu,il)
i2 = vcat(i,il,iu)
jac_sparsity = sparse(i1,i2, 1.0)
f = ODEFunction(rhs_flux!;jac_prototype=jac_sparsity);
prob_sparse = ODEProblem(f, θ0, (t0,tf), p);

#first, time a step of each



using Printf
ts = ft(3.0)
error = 0.0
last_ts_explicit = ts

while error < 1e-3
    last_ts_explicit = ts
    sol = solve(prob, Euler(), dt= ts, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0)
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    ts = ts*3.0
end

ts = ft(3.0)
error = 0.0
last_ts_implicit = ts
while error < 1e-3
    last_ts_implicit = ts
    sol = solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= ts, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0)
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    
    ts = ts*3.0
end
#reltol,abstol above 3k doesnt change result
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= last_ts_implicit, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0);

@btime solve(prob, Euler(), dt= last_ts_explicit/3.0, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0);

