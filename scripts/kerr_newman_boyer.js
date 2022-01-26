function kerr_newman(t, r, theta, phi)
{
	$cfg.rs.$default = 1;
	$cfg.r2q.$default = 0.51;
	$cfg.a.$default = -0.51;
	
    var c = 1;
    var rs = $cfg.rs;
    var r2q = $cfg.r2q;
    //dual r2q = 0.5;
    //dual a = 0.51;
    var a = $cfg.a;

    var p2 = r * r + a * a * CMath.cos(theta) * CMath.cos(theta);
    var D = r * r - rs * r + a * a + r2q * r2q;

    var dr = -p2 / D;
    var dtheta = -p2;

    var dt_1 = c * c * D / p2;
    var dtdphi_1 = -2 * c * a * CMath.sin(theta) * CMath.sin(theta) * D/p2;
    var dphi_1 = CMath.pow(a * CMath.sin(theta) * CMath.sin(theta), 2) * D/p2;

    var dphi_2 = -CMath.pow(r * r + a * a, 2) * CMath.sin(theta) * CMath.sin(theta) / p2;
    var dtdphi_2 = 2 * a * c * (r * r + a * a) * CMath.sin(theta) * CMath.sin(theta) / p2;
    var dt_2 = -a * a * c * c * CMath.sin(theta) * CMath.sin(theta) / p2;

    var dtdphi = dtdphi_1 + dtdphi_2;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = -(dt_1 + dt_2);
    ret[1 * 4 + 1] = -dr;
    ret[2 * 4 + 2] = -dtheta;
    ret[3 * 4 + 3] = -(dphi_1 + dphi_2);
    ret[0 * 4 + 3] = -dtdphi * 0.5;
    ret[3 * 4 + 0] = -dtdphi * 0.5;

    return ret;
}

kerr_newman
