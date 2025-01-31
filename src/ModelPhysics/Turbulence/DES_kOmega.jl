export KωSmagorinsky

struct KωSmagorinsky{S1,S2,S3,F1,F2,F3,C} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
end
Adapt.@adapt_structure KωSmagorinsky

# struct KωSmagorinskyModel{E1,E2,D,S}
#     k_eqn::E1 
#     ω_eqn::E2
#     Δ::D 
#     magS::S
# end 
# Adapt.@adapt_structure KωSmagorinskyModel


#Model API Constructor 
# DES{KωSmagorinsky}() = begin
#     coeffs = (β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5, C=0.15)
#     ARG = typeof(coeffs)
#     DES{KωSmagorinsky,ARG}(coeffs)
# end 

#Model Constructor using a RANS and LES model
DES{KωSmagorinsky}() = begin
    rans_model = RANS{KOmega}()  # Initialize RANS model
    les_model = LES{Smagorinsky}()    # Initialize LES model
    des_args = (C_DES=0.65, σk=0.5, σω=0.5) # DES coefficients
    X = typeof(rans_model)
    Y = typeof(les_model)
    ARG = typeof(des_args)
    DES{KωSmagorinsky, ARG}(rans_model, les_model, des_args)
end


# Function as constructor
(rans::DES{KωSmagorinsky,ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = rans.args
    KωSmagorinsky(k,omega,nut,kf,omegaf,nutf,coeffs)
end

function initialise!(turbulence::KωSmagorinsky, model::Physics{T,F,M,Tu,E,D,BI},mdotf,peqn,config) where {T,F,M,Tu,E,D,BI}

    # Initialize RANS model (for near-wall regions)
    initialise!(des.rans, mesh, config)

    # Initialize LES model (for free-stream)
    initialise!(des.les, mesh, config)

    # Compute initial DES length scale
    des_length_scale = compute_DES_lengthscale(des.rans, des.les, mesh)

    # Store the length scale inside the DES model 
    des.args = merge(des.args, (L_DES = des_length_scale,))
end

function turbulence!(des::DES, model, mesh, config)
    ν_t = compute_turbulent_viscosity(des, model, mesh)
    apply_turbulence_model!(des, model, ν_t, config)
end

#Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI},VTKWriter,name
    ) where {T,F,M,Tu<:KωSmagorinsky,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end