function func(misner_T, misner_phi, y, z)
{
	function positive_fmod(a, b)
	{	
		var r = CMath.smooth_fmod(a, b);
				
		var rlt = CMath.lt(r, 0);
				
		return CMath.select(rlt, r + b, r);
	}
	
	$cfg.phi0.$default = 0.25;
	
	var ret = [misner_T, misner_phi, y, z];
	
	ret[1] = positive_fmod(ret[1], $cfg.phi0);
	
	return ret;
}

func