function polar_to_cylindrical(t, r, theta, phi)
{
    var rp = r * CMath.sin(theta);
    var rphi = phi;
    var rz = r * CMath.cos(theta);

    return [t, rp, rphi, rz];
}

polar_to_cylindrical
