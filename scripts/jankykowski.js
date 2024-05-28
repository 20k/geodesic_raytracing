function minkowski(t, x, y, z)
{
    var t0 = CMath.select(CMath.lte(x,0),  1, -1)
    var t1 = CMath.select(CMath.lte(x,0),  -1, 1)
    var t2 = CMath.select(CMath.lte(x,0),  1, 1)
    var t3 = CMath.select(CMath.lte(x,0),  1, 1)
    
    return [t0, t1, t2, t3];
}

minkowski
