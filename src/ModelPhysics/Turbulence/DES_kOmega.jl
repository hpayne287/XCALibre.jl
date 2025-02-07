export KωSmagorinsky

struct KωSmagorinsky{S1,S2,S3,F1,F2,F3,C,M1,M2} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    rans_model::M1
    les_model::M2
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

# Really not sure this is the best way of implementing this, but as far as i can tell this is the only way to have access to both a RANS and LES model
DES{KωSmagorinsky}(nu, mesh_dev) = begin
    # rans_model = RANS{KOmega}()  # Construct RANS model
    rans_model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

    # les_model = LES{Smagorinsky}()    # Construct LES model
    les_model  = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = LES{Smagorinsky}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )
    des_args = (C_DES=0.65, σk=0.5, σω=0.5) # DES coefficients
    X = typeof(rans_model)
    Y = typeof(les_model)
    ARG = typeof(des_args)
    DES{KωSmagorinsky, ARG}(rans_model, les_model, des_args)
end


# Function as constructor
(des::DES{KωSmagorinsky,ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = des.args
    rans_model = des.rans
    les_model = des.les
    KωSmagorinsky(k,omega,nut,kf,omegaf,nutf,coeffs,rans_model,les_model)
end

function initialise(T::KωSmagorinsky, model::Physics,mdotf::FaceScalarField,peqn::ModelEquation,config) 

    #This feels like the wrong way of storing access to the LES and RANS models but works for now
    rans = T.rans_model
    les = T.les_model
    # Initialise RANS model 
    initialise(rans.turbulence, rans, mdotf, peqn, config) #Need to review what actually gets passed to the RANS and LES initialise methods
    

    # Initialise LES model 
    initialise(les.turbulence, les, mdotf, peqn, config) 

    # Compute initial DES length scale
    des_length_scale = compute_DES_lengthscale(rans, les, mesh)

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