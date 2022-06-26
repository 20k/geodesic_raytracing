function metric(v, r, theta, phi)
{
	$cfg.rs.$default = 1;
	
    var rs = $cfg.rs;

    var dv = -(1 - rs / r)
    var dv_dr = -2;
    var dtheta = r*r;
    var dphi = r*r * CMath.sin(theta) * CMath.sin(theta);

	var metric = [];
	metric.length = 16;
	
	metric[0] = dv;
	metric[1 * 4 + 0] = 0.5 * dv_dr;
	metric[0 * 4 + 1] = 0.5 * dv_dr;
	metric[2 * 4 + 2] = dtheta;
	metric[3 * 4 + 3] = dphi;
	
	return metric;
}

metric
