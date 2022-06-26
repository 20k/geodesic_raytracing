function metric(v, r, theta, phi)
{
	$cfg.rs.$default = 1;
	$cfg.a.$default = -0.5;
	
	var rs = $cfg.rs;
	var a = $cfg.a;
	
	var ct = CMath.cos(theta);
	var st = CMath.sin(theta);
	
	/*var R = r*r + a * a * ct * ct;
	var D = r*r + a * a - rs * r;
	
	var metric = [];
	metric.length = 16;
	
	var du = (1 - (rs * r) / (R*R));
	var du_dr = 2;
	var du_dphi = (2 * a * st * st / (R * R)) * (rs * r);
	var dr_dphi = -2 * a * st * st;
	var dtheta = -R * R;
	var dphi = (st * st / (R*R)) * (D * a * a * */
	
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
