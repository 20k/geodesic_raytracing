function ellis_drainhole(t, r, theta, phi)
{
    var c = 1;

    var m = 0.5;
    var n = 1;

    var alpha = CMath.sqrt(n * n - m * m);

    var pseudophi = (n / alpha) * (M_PI/2 - CMath.atan2((r - m), alpha));

    var Fp = -CMath.sqrt(1 - CMath.exp(-(2 * m/n) * pseudophi));

    var Rp = CMath.sqrt(((r - m) * (r - m) + alpha * alpha) / (1 - Fp * Fp));

    var dt1 = c*c;
    var dt2 = -Fp * Fp * c * c;
    var dp = -1;
    var dpdt = 2 * Fp * c;
    var dtheta = -Rp * Rp;
    var dphi = -Rp * Rp * CMath.sin(theta) * CMath.sin(theta);

    var ret = [];
    ret.length = 16;

    ret[0] = -dt1 - dt2;
    ret[1 * 4 + 1] = -dp;
    ret[2 * 4 + 2] = -dtheta;
    ret[3 * 4 + 3] = -dphi;
    ret[0 * 4 + 1] = -dpdt * 0.5;
    ret[1 * 4 + 0] = -dpdt * 0.5;

    return ret;
}

ellis_drainhole
