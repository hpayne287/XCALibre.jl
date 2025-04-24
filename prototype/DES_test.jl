# using Plots
using XCALibre
using Alert

using Accessors


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid1 = "flatplate_2D_lowRe.unv"
# grid1 = "bfs_unv_tet_10mm.unv"
mesh_file = joinpath(grids_dir, grid1)

# mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2025-04-20-Case20Mesh2D.unv"

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=32)

mesh_dev = mesh

U_mag = 5.4
L = 1.7
velocity = [U_mag, 0.0, 0.0]
noSlip = [0.0,0.0,0.0]
nu = 1.5e-5
Tu = 0.03

# Re = 23000
k_inlet = 3/2*(Tu*U_mag)^2
# ω_inlet = (k_inlet^0.5)/(0.09^0.25)*L
ω_inlet = 1000
nut∞ = 1e-15 

model = Physics(
    time=Transient(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=DES{Hybrid}(walls=(:wall,),blendType=MenterF1()),
    energy=Energy{Isothermal}(),
    domain=mesh_dev
)
nutBCs = (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Neumann(:top,0.0)
)


@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Neumann(:top,0.0)
)

# Example to modify/assign BCs for internal fields at API level
ransNut = assign(model.turbulence.rans.nut,nutBCs...)
@reset model.turbulence.rans.nut = ransNut

lesNut = assign(model.turbulence.les.nut, nutBCs...)
@reset model.turbulence.les.nut = lesNut

#region Define BC
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top,0.0),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top,0.0),
)

kcopy = assign(model.turbulence.rans.k, 
Dirichlet(:inlet, k_inlet),
Neumann(:outlet, 0.0),
Dirichlet(:wall, 0.0),
Neumann(:top,0.0))
@reset model.turbulence.rans.k = kcopy

ωcopy = assign(model.turbulence.rans.omega, 
Dirichlet(:inlet, ω_inlet),
Neumann(:outlet, 0.0),
Dirichlet(:wall, 1e7),
Neumann(:top,0.0))
@reset model.turbulence.rans.omega = ωcopy

#endregion

schemes = (
    U=set_schemes(divergence=Linear,time=Euler),
    p=set_schemes(time=Euler),
    y = set_schemes(gradient=Midpoint,time=Euler),
    k=set_schemes(gradient=Midpoint,time=Euler),
    omega=set_schemes(gradient=Midpoint,time=Euler)
    
)

solvers = (
    U=set_solver(
        model.momentum.U;
        solver=BicgstabSolver,
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=0.7,
        rtol=1e-4,
        atol=1e-10
    ),
    p=set_solver(
        model.momentum.p;
        solver=CgSolver,
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=0.7,
        rtol=1e-4,
        atol=1e-10
    ),
    y = set_solver(
        model.turbulence.y;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.9,
        itmax = 2000
    ),
    k = set_solver(
        model.turbulence.rans.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = set_solver(
        model.turbulence.rans.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = set_runtime(iterations=100000, time_step=0.00004, write_interval=10) #Adjust timestep to get a decent courant value

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


zeroV = [0.0,0.0,0.0]
initialise!(model.momentum.U, velocity) 
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.rans.k, k_inlet)
initialise!(model.turbulence.rans.omega, ω_inlet)
initialise!(model.turbulence.nut, nut∞)
initialise!(model.turbulence.rans.nut, nut∞)
initialise!(model.turbulence.les.nut, nut∞)

residuals = run!(model, config);

# alert("Simulation Done!!");