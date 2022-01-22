using DifferentialEquations
using OrdinaryDiffEq:
    ODEProblem,
    solve,
    ImplicitEuler,
    ImplicitMidpoint,
    Midpoint,
    Euler,
    SSPRK33,
    NLAnderson
using LinearSolve: KrylovJL_GMRES
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

i = Array(1:1:n)
iu = Array(1:1:n-1)
il = Array(2:1:n)
i1 = vcat(i,iu,il)
i2 = vcat(i,il,iu)
jac_sparsity = sparse(i1,i2, 1.0)
#=
t0 = ft(0);
tf = ft(60*60);
θ0 =  ft(0.24) .+ zeros(n)
prob = ODEProblem(rhs_flux!, θ0, (t0, tf), p);
truth = solve(prob, dt = 1.0,SSPRK33(),save_every_step =false)
u = truth.u[end]
f = ODEFunction(rhs_flux!;jac_prototype=jac_sparsity);
prob_sparse = ODEProblem(f, θ0, (t0,tf), p);
#first, time a step of each
explicit_integrator = init(prob,Euler();dt = 1.0, adaptive = false, save_every_step = false, abstol = 1e-1, reltol = 1e-1);
@btime step!(explicit_integrator)
#  23.322 μs (6 allocations: 6.70 KiB)

#atol, rtol dont affect allocations or time in meaningful way. 18x difference explicit to implicit. Converges in one Newton step?
@btime step!(init(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}); dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1));
#  458.935 μs (1583 allocations: 491.44 KiB)
@btime step!(init(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}); dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1));
#  7.526 ms (776 allocations: 1.25 MiB) - prob_sparse helps despite more allocations
@btime step!(init(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES()); dt = 1.0, save_every_step = false, adaptive = false));
#   319.132 μs (219 allocations: 221.59 KiB)#much fewer allocations compared with above
# documentation says this doesnt require a sparsity pattern. why was it using autodiff? why does it not error if we give it a sparsity pattern??


#_____________________________________________



@btime solve(prob, Euler(), dt= 1.0, save_every_step = false, adaptive = false);
#   88.004 ms (21632 allocations: 23.66 MiB)
# newton does not converge unless we increase tolerances. Again, once we are in convergent region, value of abstol, reltol doesnt change result
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1);
#    981.047 ms (349260 allocations: 696.87 MiB)
## -> need ~11x larger timesteps. consistent with a single step, close enough? 18 v 11?

@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES()), dt= 1.0, save_every_step = false, adaptive = false);
#  595.623 ms (235249 allocations: 197.68 MiB)





#Now, let's try increasing the timestep of the implicitly stepped sim. The default does not converge. Try upping tol:
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 11.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1);
#    Does not converge ^^
# Yet this does:
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 11.0, save_every_step = false, adaptive = false, reltol = 10.0, abstol = 10.0);
#  87.600 ms (32682 allocations: 63.13 MiB), and yes, this matches how long it takes the explicit integration to run with dt = 1
## Try larger:
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 110.0, save_every_step = false, adaptive = false, reltol = 10.0, abstol = 10.0);
#  9.075 ms (4692 allocations: 6.60 MiB) # yes, 10x faster than explicit with dt = 1
## and ~10x larger again:
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 1200.0, save_every_step = false, adaptive = false, reltol = 10.0, abstol = 10.0);
###  975.709 μs (1792 allocations: 815.98 KiB) again, ~90x faster than dt = 1

## Is the solution garbage?
sol = solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 1200.0, save_every_step = false, adaptive = false, reltol = 10.0, abstol = 10.0);
error = sqrt(sum((sol.u[end] .-u).^2.0)) 
# seems ok, but only the ~3 points closest to the surface have changed values in this timespan



=#
#Let's push to longer integration times, and figure out which dt for the two methods leads to ~ the same error
t0 = ft(0);
tf = ft(60*60*24);
θ0 =  ft(0.24) .+ zeros(n)
prob = ODEProblem(rhs_flux!, θ0, (t0, tf), p);

truth = solve(prob, dt = 1.0,SSPRK33(),save_every_step =false)
u = truth.u[end]
f = ODEFunction(rhs_flux!;jac_prototype=jac_sparsity);
prob_sparse = ODEProblem(f, θ0, (t0,tf), p);
##do we also need to give tol to explicit?
using Printf
ts = ft(2.0)
error = 0.0
last_ts_explicit = ts

while error < 1e-3
    sol = solve(prob, Euler(), dt= ts, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0)
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    if error < 1e-3
    last_ts_explicit = ts
    end
    ts = ts*2.0
end

ts = ft(20.0)
error = 0.0
last_ts_implicit = ts
while error < 1e-3

    sol = solve(prob_sparse,
                ImplicitEuler(autodiff=false,diff_type=Val{:central}, nlsolve = NLNewton(; κ=1//100, max_iter=10, fast_convergence_cutoff=1//5, new_W_dt_cutoff=1//5)),
                dt= ts, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0)
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    if error < 1e-3
        last_ts_implicit = ts
    end
    ts = ts*3.0
end

#= CLM style
ts = ft(20.0)
error = 0.0
last_ts_implicit = ts
while error < 1e-3

    sol = solve(prob_sparse,
                ImplicitEuler(autodiff=false,diff_type=Val{:central}, nlsolve = NLNewton(; κ=1//100, max_iter=1, fast_convergence_cutoff=1//5, new_W_dt_cutoff=1//5)),
                dt= ts, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0)
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    if error < 1e-3
        last_ts_implicit = ts
    end
    ts = ts*3.0
end
=#


ts = ft(2.0)
error = 0.0
last_ts_implicit_krylov = ts
while error < 1e-3

    sol = solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES()), dt= ts, save_every_step = false, adaptive = false, reltol = 100e-1, abstol = 100e-4) # 10x larger fails immediately.
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    if error < 1e-3
        last_ts_implicit_krylov = ts
    end
    ts = ts*3
end


ts = ft(2.0)
error = 0.0
last_ts_implicit_anderson = ts
while error < 1e-3
    sol = solve(prob,ImplicitEuler(nlsolve = NLAnderson(; κ=1//100, max_iter=10, max_history=5, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)), dt= ts, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0)
    error = sqrt(sum((sol.u[end] .-u).^2.0))
    @printf("Timestep: %lf Error: %le\n",ts, error)
    if error < 1e-3
        last_ts_implicit_anderson = ts
    end
    ts = ts*3.0
end

#reltol,abstol above 3k doesnt change result
@btime solve(prob_sparse,
             ImplicitEuler(autodiff=false,diff_type=Val{:central},nlsolve = NLNewton(; κ=1//100, max_iter=1, fast_convergence_cutoff=1//5, new_W_dt_cutoff=1//5)),
             dt= last_ts_implicit, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0);
#  4.798 ms (3193 allocations: 3.62 MiB)
#same result with maxiter = 1 or 10...so tolerances are high enough that effectively we are just taking a single newton iteration?

@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES()), dt= last_ts_implicit_krylov, save_every_step = false, adaptive = false, reltol = 100e-1, abstol = 100e-4);
#  10.597 ms (4156 allocations: 3.46 MiB)

@btime solve(prob, Euler(), dt= last_ts_explicit, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0);
#  132.311 ms (32434 allocations: 35.49 MiB)

@btime solve(prob, ImplicitEuler(nlsolve = NLAnderson(; κ=1//100, max_iter=10, max_history=5, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)), dt= last_ts_implicit_anderson, save_every_step = false, adaptive = false, reltol = 3000.0, abstol = 3000.0);
#  233.731 ms (38487 allocations: 44.01 MiB)




#=
Conclusions (?)
Implicit stepper for this problem can be pushed to about 40x faster than explicit, for the same level of error, roughly. The time per step is 18x worse. In the sim, it seems like 11x. but we can take ~450x larger steps -> 40x difference. (450/11)
Questions
Why do implicit options that are much faster per step fail for longer integrations? Newton's method most stable -> works at larger steps?
Why does a tolerance that works for shorter integration times not work for longer ones?
What does it mean to have tol > 1?
Why does it seem like the tolerance level is binary? Below a threshold, Newton does not coverge, above that threshold, changing tolerance doesnt change the solution, time to solution etc.
=#
using IncompleteLU: ilu
function incompletelu(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
  if newW === nothing || newW
    Pl = ilu(convert(AbstractMatrix,W), τ = 1e-3)
  else
    Pl = Plprev
  end
    Pl,nothing
end
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES() ,precs=incompletelu,concrete_jac=true), dt= last_ts_implicit_krylov, save_every_step = false, adaptive = false, reltol = 100e-1, abstol = 100e-4);
#  10.605 s (4365972 allocations: 3.62 GiB)# yikes
