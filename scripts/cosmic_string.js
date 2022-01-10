function cosmic_string(t, r, theta, phi)
{
    var c = 1;

    var rs = 1;

    var dt = -(1 - rs/r) * c * c;
    var dr = 1 / (1 - rs/r);
    var dtheta = r * r;

    var B = 0.3;
    var dphi = r * r * B * B * CMath.sin(theta) * CMath.sin(theta);

    return [dt, dr, dtheta, dphi];
}

cosmic_string
