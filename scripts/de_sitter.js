function de_sitter(t, r, theta, phi)
{
    var cosmo = 0.01;

    var c = 1;

    var dt = -(1 - cosmo * r * r/3) * c * c;
    var dr = 1/(1 - cosmo * r * r / 3);
    var dtheta = r * r;
    var dphi = r * r * CMath.sin(theta) * CMath.sin(theta);

    return [dt, dr, dtheta, dphi];
}

de_sitter
