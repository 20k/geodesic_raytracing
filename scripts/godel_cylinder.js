function metric(t, r, phi, z)
{    
	$cfg.a.$default = 1000
    
    var a = $cfg.a
    
    var dt = -1;
    var dr = 1/(1 + CMath.pow(r/(2*a), 2))
    var dphi = r*r * (1 - CMath.pow(r/(2 * a), 2))
    var dz = 1
    var dt_dphi = -2 * r * r * (1/(CMath.sqrt(2) * a))
    
    var ret = [];
    ret.length = 16;
    ret[0] = dt;
    ret[1 * 4 + 1] = dr;
    ret[2 * 4 + 2] = dphi;
    ret[3 * 4 + 3] = dz;
    ret[0 * 4 + 2] = 0.5 * dt_dphi;
    ret[2 * 4 + 0] = 0.5 * dt_dphi;
    
    return ret;
}

metric
