function ingoing_ef_to_polar(v, r, theta, phi)
{
	var rs = $cfg.rs;
	
    /*var r_star = r + rs * CMath.log(CMath.fabs((r / rs) - 1));
    
	var t = v - r_star;*/
	
	//var v = t + r + rs * CMath.log(fabs(r - rs));

	var t = v - (r + rs * CMath.log(CMath.fabs(r - rs)))

    return [t, r, theta, phi];
}

ingoing_ef_to_polar
