function metric(v, r, theta, phi)
{
	//http://www.scholarpedia.org/article/Kerr-Newman_metric (47), with signs flipped due to signature
	$cfg.rs.$default = 1;
	$cfg.a.$default = -0.5;
	
	var rs = $cfg.rs;
	var a = $cfg.a;
	
	var ct = CMath.cos(theta);
	var st = CMath.sin(theta);
	
	var R2 = r*r + a * a * ct * ct;
	var D = r*r + a * a - rs * r;
	
	var metric = [];
	metric.length = 16;
	
	var dv = (1 - (rs * r) / R2);
	var dv_dr = -2;
	var dv_dphi = (2 * a * st * st / R2) * (rs * r);
	var dr_dphi = 2 * a * st * st;
	var dtheta = -R2;
	var dphi = (st * st / R2) * (D * a * a * st * st - CMath.pow(a * a + r*r, 2));
	
	///v, r, theta, phi
	metric[0] = -dv;
	metric[1 * 4 + 0] = -0.5 * dv_dr;
	metric[0 * 4 + 1] = -0.5 * dv_dr;
	
	metric[3 * 4 + 0] = -0.5 * dv_dphi;
	metric[0 * 4 + 3] = -0.5 * dv_dphi;
	
	metric[1 * 4 + 3] = -0.5 * dr_dphi;
	metric[3 * 4 + 1] = -0.5 * dr_dphi;
	
	metric[2 * 4 + 2] = -dtheta;
	metric[3 * 4 + 3] = -dphi;
	
	return metric;
	
	/*var l = [1, 0, 0, a * st * st];
	
	var g0ab = [-1, 1, 0, 0,
	            1, 0, 0, a * st*st,
				0, 0, r*r + a*a * ct*ct, 0,
				0, a * st*st, 0, (r*r + a*a) * st*st];
	

	var metric = [];
	metric.length = 16;
	
	for(var i=0; i < 4; i++)
	{
		for(var j=0; j < 4; j++)
		{
			metric[i * 4 + j] = g0ab[i * 4 + j] + (rs * r / (r*r + a*a * ct*ct)) * l[i] * l[j];
		}
	}
		
	return metric;*/
}

metric
