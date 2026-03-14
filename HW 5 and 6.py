import sympy as sp

t, x, y, z = sp.symbols('t x y z')
vx, vy, vz = sp.symbols('vx vy vz')

q, m = sp.symbols('q m')
epsilon0, mu0 = sp.symbols('epsilon0 mu0')
B0 = sp.symbols('B0')

# position and velocity vectors
r = sp.Matrix([x, y, z])
v = sp.Matrix([vx, vy, vz])


def divergence(F):
    return sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)

def curl(F):
    return sp.Matrix([
        sp.diff(F[2], y) - sp.diff(F[1], z),
        sp.diff(F[0], z) - sp.diff(F[2], x),
        sp.diff(F[1], x) - sp.diff(F[0], y)
    ])


def verify_maxwell(E, B):

    divE = divergence(E)
    divB = divergence(B)

    curlE = curl(E)
    curlB = curl(B)

    dE_dt = E.diff(t)
    dB_dt = B.diff(t)

    # Charge density from Gauss's law
    rho_fields = epsilon0 * divE

    # Current density from Ampere-Maxwell
    J_fields = (1/mu0) * (curlB - mu0*epsilon0*dE_dt)

    # Maxwell residuals
    gauss_residual = divE - rho_fields/epsilon0
    magnetic_residual = divB
    faraday_residual = curlE + dB_dt
    ampere_residual = curlB - mu0*epsilon0*dE_dt - mu0*J_fields

    return {
        "rho_fields": rho_fields,
        "J_fields": J_fields,
        "gauss_residual": sp.simplify(gauss_residual),
        "magnetic_residual": sp.simplify(magnetic_residual),
        "faraday_residual": sp.simplify(faraday_residual),
        "ampere_residual": sp.simplify(ampere_residual)
    }


def compute_charge_current_from_f(f):

    rho_f = q * sp.integrate(f, (vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo))

    J_fx = q * sp.integrate(vx*f, (vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo))
    J_fy = q * sp.integrate(vy*f, (vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo))
    J_fz = q * sp.integrate(vz*f, (vx,-sp.oo,sp.oo),(vy,-sp.oo,sp.oo),(vz,-sp.oo,sp.oo))

    J_f = sp.Matrix([J_fx, J_fy, J_fz])

    return rho_f, J_f

def verify_vlasov(E, B, f):

    df_dt = sp.diff(f, t)

    grad_x_f = sp.Matrix([
        sp.diff(f, x),
        sp.diff(f, y),
        sp.diff(f, z)
    ])

    v_dot_gradxf = v.dot(grad_x_f)

    v_cross_B = v.cross(B)

    grad_v_f = sp.Matrix([
        sp.diff(f, vx),
        sp.diff(f, vy),
        sp.diff(f, vz)
    ])

    lorentz_term = (q/m) * (E + v_cross_B).dot(grad_v_f)

    vlasov_residual = sp.simplify(df_dt + v_dot_gradxf + lorentz_term)

    return vlasov_residual

print("\n=============================")
print("Example 1: Uniform Plasma at Rest")
print("=============================")

E1 = sp.Matrix([0,0,0])
B1 = sp.Matrix([0,0,0])

f1 = sp.Function('f')(vx,vy,vz)

maxwell1 = verify_maxwell(E1,B1)

rho_f1, J_f1 = compute_charge_current_from_f(f1)

vlasov1 = verify_vlasov(E1,B1,f1)

print("\nMaxwell Residuals:")
print("Gauss:", maxwell1["gauss_residual"])
print("Magnetic:", maxwell1["magnetic_residual"])
print("Faraday:", maxwell1["faraday_residual"])
print("Ampere:", maxwell1["ampere_residual"])

print("\nCharge/Current from Fields:")
print("rho =", maxwell1["rho_fields"])
print("J =", maxwell1["J_fields"])

print("\nCharge/Current from Distribution:")
print("rho_f =", rho_f1)
print("J_f =", J_f1)

print("\nVlasov Residual:")
print(vlasov1)


print("\n=============================")
print("Example 2: Uniform Magnetic Field")
print("=============================")

E2 = sp.Matrix([0,0,0])
B2 = sp.Matrix([0,0,B0])

f2 = sp.Function('f')(vx,vy,vz)

maxwell2 = verify_maxwell(E2,B2)

rho_f2, J_f2 = compute_charge_current_from_f(f2)

vlasov2 = verify_vlasov(E2,B2,f2)

print("\nMaxwell Residuals:")
print("Gauss:", maxwell2["gauss_residual"])
print("Magnetic:", maxwell2["magnetic_residual"])
print("Faraday:", maxwell2["faraday_residual"])
print("Ampere:", maxwell2["ampere_residual"])

print("\nCharge/Current from Fields:")
print("rho =", maxwell2["rho_fields"])
print("J =", maxwell2["J_fields"])

print("\nCharge/Current from Distribution:")
print("rho_f =", rho_f2)
print("J_f =", J_f2)

print("\nVlasov Residual:")
print(vlasov2)

print("\nVerification complete.")