import numpy as np

def strike_slip(D: float=0.9144, t: float=0.0119,
    E1: float=210e6, s1: float=490e3, s2: float=531e3, e2:float = 0.04,
    tu: float=40.5, qu: float=318.6, xu: float=11.4e-3,
    Df: float=0.9144, psi: float=0, beta: float=60):

    # Pipeline geometry
    Rm = (D - t) / 2
    As = np.pi * (D**2 - (D-2*t)**2) / 4
    I = np.pi * (D**4 - (D-2*t)**4) / 64

    # Pipeline steel
    e1 = s1 / E1
    E2 = (s2 - s1) / (e2 - e1)

    # Fault Displacement
    Dh = Df * np.cos(np.deg2rad(psi))
    Dx = Dh * np.sin(np.deg2rad(beta))
    Dy = Dh * np.cos(np.deg2rad(beta))
    Dz = Df * np.sin(np.deg2rad(psi))


    # Step 1:
    k = qu / xu
    l = (k / (4*E1*I))**0.25
    Cr = 2 * l * E1*I


    # Step 2:
    E = E1
    d = Dy / 2

    for iE in range(100):
        a0 = 24 * E*I * d * Cr
        a1 = 24 * E*I * d * Cr * l
        a3 = 12 * E*I * qu
        a4 = 5 * qu * Cr
        a5 = qu * Cr * l

        Lc = 100
        for iL in range(100):
            fL = a5*Lc**5 + a4*Lc**4 + a3*Lc**3 - a1*Lc - a0
            dfL = 5*a5*Lc**4 + 4*a4*Lc**3 + 3*a3*Lc**2 - a1
            dLc = fL/dfL
            Lc -= dLc
            if np.abs(dLc) < 1e-6:
                break
        if iE == 0:
            Lc0 = Lc

        MA = (24 * E*I * d * Cr - qu * Cr * Lc**4) /\
            (24 * E*I * Lc + 8 * Cr * Lc**2)
        VA = (24 * E*I * d * Cr - 12 * E*I * qu * Lc**3 - 5 * qu * Cr * Lc**4) /\
            (24 * E*I * Lc**2 + 8 * Cr * Lc**3)
        VB = (24 * E*I * d * Cr + 12 * E*I * qu * Lc**3 + 3 * qu * Cr * Lc**4) /\
            (24 * E*I * Lc**2 + 8 * Cr * Lc**3)
        xmax = VB / qu
        Mmax = VB * xmax - qu * xmax**2 / 2


        # Step 3:
        DLreq = Dx
        DLav_el = (s1**2 * As) / (E1 * tu)
        if DLreq <= DLav_el:
            sa = np.sqrt(E1 * tu * DLreq / As)
        else:
            sa = (s1 * (E1 - E2) +\
                np.sqrt(s1**2 * (E2**2 - E1 * E2) + E1**2 * E2 * DLreq * tu / As)) / E1
        Fa = sa * As
        Lanch = Fa / tu


        # Step 4:
        eb_I = Mmax * D / (2 * E * I)
        if Fa == 0:
            eb = eb_I
        else:
            R = Fa / qu
            eb_II = qu * D / (2 * Fa)
            eb = 1 / (1/eb_I + 1/eb_II)


        # Step 5:
        ea = 0
        for iF in range(100):
            cosf1 = (e1 - ea) / eb
            cosf2 = (e1 + ea) / eb
            f1 = np.pi if cosf1 <= -1 else np.arccos(cosf1) if cosf1 < 1 else 0
            f2 = np.pi if cosf2 <= -1 else np.arccos(cosf2) if cosf2 < 1 else 0
            if eb * np.sin(f1) <= -0.01:
                df1 = 1/(eb * np.sin(f1))
            elif eb * np.sin(f1) <= 0:
                df1 = -100
            elif eb * np.sin(f1) <= 0.01:
                df1 = +100
            else:
                df1 = 1/(eb * np.sin(f1))
            if eb * np.sin(f2) <= -0.01:
                df2 = -1/(eb * np.sin(f2))
            elif eb * np.sin(f2) <= 0:
                df2 = -100
            elif eb * np.sin(f2) <= 0.01:
                df2 = -100
            else:
                df2 = -1/(eb * np.sin(f2))
            F = 2 * Rm * t * (E1 * np.pi * ea - (E1 - E2)*(f1 + f2)*ea \
                + (E1 - E2)*(f1 - f2)*e1 \
                - (E1 - E2)*(np.sin(f1) - np.sin(f2))*eb)
            dF = 2 * Rm * t * (E1 * np.pi - (E1 - E2)*(f1 + f2) \
                - (E1 - E2)*(df1 + df2)*ea + (E1 - E2)*(df1 - df2)*e1 \
                - (E1 - E2)*(df1 * np.cos(f1) - df2 * np.cos(f2))*eb)
            dea = (F - Fa) / dF
            ea -= dea
            if np.abs(dea) < 1e-6:
                break


        # Step 6:
        M = 2 * Rm**2 * t * (E1 * np.pi * eb / 2 - (E1 - E2)*(np.sin(f1) - np.sin(f2))*ea \
            + (E1 - E2)*(np.sin(f1) + np.sin(f2))*e1 - (E1 - E2)*(f1 + f2)*eb / 2 \
            - (E1 - E2)*(np.sin(2*f1) + np.sin(2*f2))*eb / 4)
        Enew = (M * D) / (2 * I * eb_I)
        if abs(E - Enew) > 1e-3:
            E = Enew
        else:
            break


    # Returns
    emax = ea + eb
    emin = ea - eb
    smax = emax * E1 if np.abs(emax) < e1 \
        else np.sign(emax) * (s1 + (np.abs(emax) - e1) * E2)
    smin = emin * E1 if np.abs(emin) < e1 \
        else np.sign(emin) * (s1 + (np.abs(emin) - e1) * E2)

    return ea, eb, emax, emin, smax, smin, F, M, E, Lc, Lanch