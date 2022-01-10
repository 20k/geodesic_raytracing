function cylindrical_to_polar(t, p, phi, z)
{
    var rr = CMath.sqrt(p * p + z * z);
    var rtheta = CMath.atan2(p, z);
    var rphi = phi;

    return [t, rr, rtheta, rphi];
}

cylindrical_to_polar
