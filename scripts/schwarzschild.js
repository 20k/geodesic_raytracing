function metric(t, r, theta, phi)
{
    var rs = 1;
    var c = 1;

    var dt = -(1 - rs / r) * c * c;
    var dr = 1/(1 - rs / r);
    var dtheta = r*r;
    var dphi = r*r * CMath.sin(theta) * CMath.sin(theta);

    return [dt, dr, dtheta, dphi];
}

metric
