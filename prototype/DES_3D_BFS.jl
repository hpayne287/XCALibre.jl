# using Plots
using XCALibre
using Alert
using StaticArrays
using Accessors

mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2025-05-02-Case31Mesh2D.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=32)

mesh_dev = mesh

reh = 5100
nu = 1e-5
h = 0.1
Umag = reh*nu/h
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]

νR = 5
Tu = 0.05
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
nut∞ = 1e-15 

model = Physics(
    time=Transient(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=DES{Hybrid}(walls=(:wall1,:wall2,:side),blendType=MenterF1()),
    energy=Energy{Isothermal}(),
    domain=mesh_dev
)



nutBCs = (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    # Dirichlet(:wall,0.0),
    Dirichlet(:wall1, 0.0),
    Dirichlet(:wall2, 0.0),
    Dirichlet(:side, 0.0),
    Dirichlet(:top,0.0),
    # Neumann(:left, 0.0),
    # Neumann(:right, 0.0)
)

ransNut = assign(model.turbulence.rans.nut,nutBCs...)
@reset model.turbulence.rans.nut = ransNut

lesNut = assign(model.turbulence.les.nut, nutBCs...)
@reset model.turbulence.les.nut = lesNut

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    # Dirichlet(:wall,0.0),
    Dirichlet(:wall1, 0.0),
    Dirichlet(:wall2, 0.0),
    Dirichlet(:side, 0.0),
    Dirichlet(:top,0.0),
    # Neumann(:left, 0.0),
    # Neumann(:right, 0.0)
)

#region Define BC
@assign! model momentum U (
    # DirichletFunction(:inlet, blasius_inlet),
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Dirichlet(:wall, [0.0,0.0,0.0]),
    Dirichlet(:wall1, [0.0, 0.0, 0.0]),
    Dirichlet(:wall2, [0.0, 0.0, 0.0]),
    Dirichlet(:side, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0]),
    # Neumann(:top,0.0),
    # Neumann(:left, 0.0),
    # Neumann(:right, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall1, 0.0),
    Neumann(:wall2, 0.0),
    Neumann(:side, 0.0),
    Neumann(:top,0.0),
    # Neumann(:left, 0.0),
    # Neumann(:right, 0.0)
)

kcopy = assign(model.turbulence.rans.k, 
Dirichlet(:inlet, k_inlet),
Neumann(:outlet, 0.0),
# Dirichlet(:wall, 0.0),
Dirichlet(:wall1, 0.0),
Dirichlet(:wall2, 0.0),
Dirichlet(:side, 0.0),
Dirichlet(:top,0.0),
# Neumann(:left, 0.0),
# Neumann(:right, 0.0)
)
@reset model.turbulence.rans.k = kcopy

ωcopy = assign(model.turbulence.rans.omega, 
Dirichlet(:inlet, ω_inlet),
Neumann(:outlet, 0.0),
# Dirichlet(:wall, 1e6),
Dirichlet(:wall1, 1e6),
Dirichlet(:wall2, 1e6),
Dirichlet(:side, 1e6),
Dirichlet(:top, 1e6),
# Neumann(:left, 0.0),
# Neumann(:right, 0.0)
)
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
        solver      = CgSolver, 
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.9,
        itmax = 5000
    ),
    k = set_solver(
        model.turbulence.rans.k;
        solver      = BicgstabSolver, 
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = set_solver(
        model.turbulence.rans.omega;
        solver      = BicgstabSolver, 
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = set_runtime(iterations=10000, time_step=0.0005, write_interval=10) 

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)
GC.gc()

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