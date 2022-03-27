function spinning_cosmic_string(t, p, phi, z)
{
    var c = 1;

	$cfg.a.$default = 0.01;
	$cfg.k.$default = 0.98;

    ///spin
    var a = $cfg.a;
    ///deficit angle is (1 - k) * 2pi, aka the segment cut out of a circle
    var k = $cfg.k;
    ///a = 0, k = 1 = minkowski

    var dt = -1;
    var dtdphi = 2 * a;
    var dphi1 = a * a;

    var dz = 1;
    var dp = 1;

    var dphi2 = k*k * p*p;

    var ret = [];
    ret.length = 16;

    ret[0] = dt;
    ret[1 * 4 + 1] = dp;
    ret[2 * 4 + 2] = dphi1 + dphi2;
    ret[3 * 4 + 3] = dz;

    ret[0 * 4 + 1] = 0.5 * dtdphi;
    ret[1 * 4 + 0] = 0.5 * dtdphi;

    return ret;
}

spinning_cosmic_string
