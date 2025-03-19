function get_cross_diff(rans,des,model,config)
    (; ω_eqn, ∇k, ∇ω) = des
    (; σd) = model.turbulence.coeffs
    (; k, omega, kf, omegaf, nutf) = model.turbulence

    dkdomegadx = get_source(ω_eqn, 2)

    interpolate!(kf, k, config)
    correct_boundaries!(nutf, k, k.BCs, time, config)
    interpolate!(omegaf, omega, config)
    correct_boundaries!(nutf, omega, omega.BCs, time, config)
    grad!(∇ω, omegaf, omega, omega.BCs, time, config)
    grad!(∇k, kf, k, k.BCs, time, config)
    inner_product!(dkdomegadx, ∇k, ∇ω, config)
    @. dkdomegadx.values = max((σd / omega.values) * dkdomegadx.values, 0.0)

    return dkdomegadx
end

function get_cross_diff(rans::KOmegaLKE,des,model,config)
    (; ω_eqn) = rans
    dkdomegadx = get_source(ω_eqn,2)

    return dkdomegadx
end

function set_eddy_viscosity(model::Physics) 
    (; nut, rans, les) = model.turbulence
    @. rans.nut.values = nut.values
    @. les.nut.values = nut.values
end

"""
    blend!(blendType, des::HybridModel, model::Physics, config)

Set the values of the `blendWeight` field according to the method specified by `blendType`

### Input
- `blendType` -- Blending method <: AbstractBlendingMethod
- `des` -- turbulence model.
- `model` -- Physics model defined by user.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.
"""
function blend!() end #Dummy for documentation

function blend!(blendType::MenterF1, des::HybridModel, model::Physics, config)
    (; rho) = model.fluid
    (; k, omega, nut, blendWeight, CDkw, y, rans) = model.turbulence
    (; βstar, σω2) = model.turbulence.coeffs
    (; ω_eqn) = des

    dkdomegadx = get_cross_diff(rans,des,model,config)

    @. CDkw.values = max((2 * rho.values * σω2 * (1 / omega.values) * dkdomegadx.values), 10e-20)
    @. blendWeight.values = tanh(min(max(sqrt(k.values) / (βstar * y.values * omega.values),
            (500 * nut.values) / (y.values^2 * omega.values)),
        (4 * rho.values * σω2 * k.values) / (CDkw.values * y.values^2))^4)

end

function blend_nut!(nut, blend, nutRANS, nutLES)
    @. nut.values = (blend.values * nutRANS.values) + ((1 - blend.values) * nutLES.values)
end