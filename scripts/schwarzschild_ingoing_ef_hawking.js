function metric(v, r, theta, phi)
{
	$cfg.rs.$default = 1;
    $cfg.lifetime.$default = 1000;
	
    //https://arxiv.org/pdf/2103.08340.pdf
    var rs_base = $cfg.rs;
    //rs = 2 * M0
    
    var M0 = rs_base/2;
    
    var v_lifetime = $cfg.lifetime; ///real values are 10^70
    
    var k_squiggle = M0 * M0 * M0 / v_lifetime;
    var k_dash = 2 * CMath.pow(k_squiggle, 1/3.);
    
    var negative_branch = k_dash * CMath.pow(v_lifetime - v, 1/3.)
    
    var rs_v = CMath.select(CMath.lte(v, v_lifetime), negative_branch, 0) 

    var dv = -(1 - rs_v / r)
    var dv_dr = 2;
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
