import sympy as sp

t,x,y,z = sp.symbols('t x y z')
vx,vy,vz = sp.symbols('vx vy vz')
q,m,eps0,mu0,B0 = sp.symbols('q m eps0 mu0 B0')

v = sp.Matrix([vx,vy,vz])


def div(F):
    return sp.diff(F[0],x)+sp.diff(F[1],y)+sp.diff(F[2],z)

def curl(F):
    return sp.Matrix([
        sp.diff(F[2],y)-sp.diff(F[1],z),
        sp.diff(F[0],z)-sp.diff(F[2],x),
        sp.diff(F[1],x)-sp.diff(F[0],y)
    ])


def maxwell(E,B):

    divE, divB = div(E), div(B)
    curlE, curlB = curl(E), curl(B)

    dE = E.diff(t)

    rho = eps0*divE
    J = (1/mu0)*(curlB - mu0*eps0*dE)

    return {
        "Gauss": sp.simplify(divE-rho/eps0),
        "divB": sp.simplify(divB),
        "Faraday": sp.simplify(curlE + B.diff(t)),
        "Ampere": sp.simplify(curlB - mu0*eps0*dE - mu0*J),
        "rho": rho,
        "J": J
    }

def kinetic(f):

    rho_f = q*sp.integrate(f,(vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo))

    J = sp.Matrix([
        q*sp.integrate(vx*f,(vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo)),
        q*sp.integrate(vy*f,(vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo)),
        q*sp.integrate(vz*f,(vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo))
    ])

    return rho_f, J

def vlasov(E,B,f):

    dfdt = sp.diff(f,t)

    gradx = sp.Matrix([sp.diff(f,x),sp.diff(f,y),sp.diff(f,z)])
    gradv = sp.Matrix([sp.diff(f,vx),sp.diff(f,vy),sp.diff(f,vz)])

    return sp.simplify(
        dfdt
        + v.dot(gradx)
        + (q/m)*(E + v.cross(B)).dot(gradv)
    )

def run_case(E,B,f,title):

    print("\n---",title,"---")

    M = maxwell(E,B)
    rho_f,J_f = kinetic(f)
    V = vlasov(E,B,f)

    print("Gauss:",M["Gauss"])
    print("divB:",M["divB"])
    print("Faraday:",M["Faraday"])
    print("Ampere:",M["Ampere"])

    print("\nrho(fields) =",M["rho"])
    print("rho(f) =",rho_f)

    print("\nJ(fields) =",M["J"])
    print("J(f) =",J_f)

    print("\nVlasov residual =",V)

E1 = sp.Matrix([0,0,0])
B1 = sp.Matrix([0,0,0])
f1 = sp.Function('f')(vx,vy,vz)

run_case(E1,B1,f1,"Example 1: Plasma at Rest")

E2 = sp.Matrix([0,0,0])
B2 = sp.Matrix([0,0,B0])
f2 = sp.Function('f')(vx,vy,vz)

run_case(E2,B2,f2,"Example 2: Uniform Magnetic Field")

print("\nVerification complete.")